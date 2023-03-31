from django.db import models, transaction
import numpy as np
from django.conf import settings
from gtm_hit.misc import geometry
from gtm_hit.models import Annotation, Annotation2DView, MultiViewFrame, Person, View
from django.core.exceptions import ObjectDoesNotExist
from ipdb import set_trace
def find_closest_annotations_to(person_id,frame_id):
    # Check if frame_id is an integer
    if not isinstance(frame_id, int):
        raise ValueError("Frame ID must be an integer.")
    if not isinstance(person_id, int):
        raise ValueError("Person ID must be an integer.")
    try:
        #set_trace()
        next_annotation = Annotation.objects.filter(person__person_id=person_id, frame__frame_id__gt=frame_id,frame__undistorted=settings.UNDISTORTED_FRAMES).order_by('frame__frame_id').first()
        last_annotation = Annotation.objects.filter(person__person_id=person_id, frame__frame_id__lte=frame_id,frame__undistorted=settings.UNDISTORTED_FRAMES).order_by('frame__frame_id').last()
        
        if last_annotation is None or next_annotation is None:
            raise ObjectDoesNotExist

    except ObjectDoesNotExist:
        raise ValueError(f"No next annotation found for person {person_id} after frame {frame_id}.")

    return last_annotation, next_annotation

def save_2d_views(annotation):
    for i in range(settings.NB_CAMS):
        try:
            view, _ = View.objects.get_or_create(view_id=i)
            try:
                cuboid = geometry.get_cuboid2d_from_annotation(
                    annotation, settings.CALIBS[i], settings.UNDISTORTED_FRAMES)
                p1, p2 = geometry.get_bounding_box(cuboid)
            except ValueError:
                cuboid = None
                p1 = [-1, -1]
                p2 = [-1, -1]
            # Set the cuboid points for the annotation2dview object
            # set_trace()
            try:
                annotation2dview = Annotation2DView.objects.get(
                    view=view,
                    annotation=annotation)
            except Annotation2DView.DoesNotExist:
                annotation2dview = Annotation2DView(
                    view=view,
                    annotation=annotation
                )
            annotation2dview.x1 = p1[0]
            annotation2dview.y1 = p1[1]
            annotation2dview.x2 = p2[0]
            annotation2dview.y2 = p2[1]
            if cuboid is not None:
                annotation2dview.set_cuboid_points_2d(cuboid)
            annotation2dview.save()
        except Exception as e:
            print(e)
            return False

def interpolate_between_annotations(person_id,frame_id1,frame_id2):

    annotation1 = Annotation.objects.get(person__person_id=person_id, frame__frame_id=frame_id1,frame__undistorted=settings.UNDISTORTED_FRAMES)
    annotation2 = Annotation.objects.get(person__person_id=person_id, frame__frame_id=frame_id2,frame__undistorted=settings.UNDISTORTED_FRAMES)
    save_2d_views([annotation1, annotation2]) # Save 2D views for the two annotations
    # Check if frame_id1 and frame_id2 are integers
    if not isinstance(frame_id1, int) or not isinstance(frame_id2, int):
        raise ValueError("Frame IDs must be integers.")
    num_interpolations = frame_id2 - frame_id1 -1

    for j in range(1, num_interpolations + 1):
        interpolated_frame_id = frame_id1 + j
        interpolated_frame = MultiViewFrame.objects.get_or_create(frame_id=interpolated_frame_id, undistorted=settings.UNDISTORTED_FRAMES, worker_id=annotation1.frame.worker_id)[0]

        # Try to get the existing annotation for the given person and frame
        try:
            interpolated_annotation = Annotation.objects.get(person__person_id=person_id, frame=interpolated_frame)
        except Annotation.DoesNotExist:
            # If the annotation does not exist, create a new one
            interpolated_annotation = Annotation()
            interpolated_annotation.frame = interpolated_frame
            interpolated_annotation.person = annotation1.person

        interpolated_annotation.rectangle_id = f"interpolated-{annotation1.rectangle_id}-{annotation2.rectangle_id}-{j}"
        interpolated_annotation.creation_method = "interpolated"
        interpolated_annotation.validated = False
        interpolated_annotation.rotation_theta = np.interp(interpolated_frame_id, [frame_id1, frame_id2], [annotation1.rotation_theta, annotation2.rotation_theta])
        interpolated_annotation.Xw = np.interp(interpolated_frame_id, [frame_id1, frame_id2], [annotation1.Xw, annotation2.Xw])
        interpolated_annotation.Yw = np.interp(interpolated_frame_id, [frame_id1, frame_id2], [annotation1.Yw, annotation2.Yw])
        interpolated_annotation.Zw = np.interp(interpolated_frame_id, [frame_id1, frame_id2], [annotation1.Zw, annotation2.Zw])
        interpolated_annotation.object_size_x = np.interp(interpolated_frame_id, [frame_id1, frame_id2], [annotation1.object_size_x, annotation2.object_size_x])
        interpolated_annotation.object_size_y = np.interp(interpolated_frame_id, [frame_id1, frame_id2], [annotation1.object_size_y, annotation2.object_size_y])
        interpolated_annotation.object_size_z = np.interp(interpolated_frame_id, [frame_id1, frame_id2], [annotation1.object_size_z, annotation2.object_size_z])
        #set_trace()
        try:
            interpolated_annotation.save()
        except Exception as e:
            raise (f"Could not save interpolated annotation for person {person_id} for frame {interpolated_frame_id}.")
        save_2d_views(interpolated_annotation)
    return f"Interpolated annotations for person {person_id} between frames {frame_id1} and {frame_id2} have been created."

def interpolate_until_next_annotation(person_id,frame_id):

    #set_trace()
    annotation1, annotation2 = find_closest_annotations_to(person_id,frame_id)
    frame_id1 = annotation1.frame.frame_id
    frame_id2 = annotation2.frame.frame_id
    return interpolate_between_annotations(person_id,frame_id1,frame_id2)


def propagation_filter(arguments,options,frame_id):
    if options["propagate"] == 'future':
        arguments['frame__frame_id__gte'] = frame_id
    elif options["propagate"] == 'past':
        arguments['frame__frame_id__lte'] = frame_id
    else:
        arguments['frame__frame_id'] = frame_id
    return arguments


def get_next_available_id():
    max_id = Person.objects.aggregate(models.Max('pk'))['pk__max']
    return max_id + 1 if max_id is not None else 1


def change_annotation_id_propagate(old_id, new_id, frame_id, options):
    with transaction.atomic():  # Start a transaction to ensure data consistency
        try:
            next_id = get_next_available_id()
            filterargs={'person__person_id': new_id, 'frame__undistorted':settings.UNDISTORTED_FRAMES}
            filterargs = propagation_filter(filterargs,options,frame_id)
            # Find all conflicting future annotations
            annotation_conflicts = Annotation.objects.filter(**filterargs).order_by('frame__frame_id')
            #set_trace()
            if options["conflicts"] == "assign_new":
                for conflict in annotation_conflicts:
                    # Update the conflicting annotation's person with the new unique ID
                    person_to_replace = conflict.person
                    person_to_replace.person_id = next_id
                    person_to_replace.save()

                    # Update the related Annotation object
                    conflict.person = person_to_replace
                    conflict.save()
            elif options["conflicts"] == "delete":
                for conflict in annotation_conflicts:
                    conflict.delete()

            # Find all target future annotations
            filterargs["person__person_id"]=old_id
            target_future_annotations = Annotation.objects.filter(**filterargs).order_by('frame__frame_id')

            for annotation in target_future_annotations:
                target_future_person = annotation.person
                target_future_person.person_id = new_id
                target_future_person.save()

                # Update the related Annotation object
                annotation.person = target_future_person
                annotation.save()

            return True
        except Exception as e:
            print(e)
            return False


def get_annotation2dviews_for_frame_and_person(frame_id, person_id):
    # Get the MultiViewFrame object for the given frame_id
    #frame = MultiViewFrame.objects.get(frame_id=frame_id)

    # Calculate the range of frame_ids for 5 frames before and 5 frames after the given frame
    frame_id_start = max(1, frame_id - 100)
    frame_id_end = frame_id + 100

    # Get the Person object for the given person_id
    person = Person.objects.get(person_id=person_id)

    # Filter the Annotation2DView objects using the calculated frame range and the Person object
    annotation2dviews = Annotation2DView.objects.filter(
        annotation__frame__frame_id__gte=frame_id_start,
        annotation__frame__frame_id__lte=frame_id_end,
        annotation__person=person,
        annotation__frame__undistorted = settings.UNDISTORTED_FRAMES
    )
    #set_trace()
    camtrackviews = {}
    for annotation2dview in annotation2dviews:
        vid = annotation2dview.view.view_id
        try:
            base_point = annotation2dview.cuboid_points_2d[8]
        except:
            continue
        if vid not in camtrackviews:
            camtrackviews[vid] = []
        camtrackviews[vid].append(
            (annotation2dview.annotation.frame.frame_id, base_point))
    for view_id in camtrackviews:
        camtrackviews[view_id].sort()
    return camtrackviews