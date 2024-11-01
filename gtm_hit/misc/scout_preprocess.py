import cv2
import numpy as np
from IPython.display import clear_output
import time
import json
import uuid
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person,Dataset
from django.conf import settings
from gtm_hit.misc.db import save_2d_views, save_2d_views_bulk
import os
from ipdb import set_trace
import os.path as osp

from pathlib import Path
from typing import Optional
import pickle
from tqdm import tqdm

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def transfer_complete_annotations(dataset_name:str, 
                                  worker_from:str="SYNC17APR0908",
                                  worker_to:str="IVANA"):
    
    dataset = Dataset.objects.get(name=dataset_name)
    worker = Worker.objects.get(workerID=worker_from)

    persons_complete = Person.objects.filter(
        dataset=dataset, 
        annotation_complete=True, 
        worker=worker)

    for person_complete in persons_complete:
        person_complete_worker = person_complete.worker
        other_worker = Worker.objects.get(workerID=worker_to)
        complete_annotations = Annotation.objects.filter(person=person_complete)

        print("***")

        try:
            person_other_worker = Person.objects.get(
                person_id=person_complete.person_id, 
                worker=other_worker, 
                dataset=dataset)
            
        except Person.DoesNotExist:
            continue

        print(f"Syncing {person_complete.person_id} from {person_complete.worker.workerID} to {other_worker.workerID}...")

        for complete_annotation in complete_annotations:
            try:
                target_annotation = Annotation.objects.get(
                    person=person_other_worker, 
                    frame__frame_id=complete_annotation.frame.frame_id, 
                    frame__undistorted=complete_annotation.frame.undistorted, 
                    frame__worker=other_worker, 
                    frame__dataset=dataset
                    )
            except Annotation.DoesNotExist:
                target_annotation = Annotation()

                frame, _ = MultiViewFrame.objects.get_or_create(
                    frame_id=complete_annotation.frame.frame_id, 
                    undistorted=complete_annotation.frame.undistorted, 
                    worker=other_worker, 
                    dataset=dataset)
                
                target_annotation.frame = frame
                target_annotation.person = person_other_worker

            target_annotation.rectangle_id = complete_annotation.rectangle_id
            target_annotation.creation_method = f"sync_{person_complete_worker.workerID}__{complete_annotation.creation_method}"
            target_annotation.validated = complete_annotation.validated
            target_annotation.rotation_theta = complete_annotation.rotation_theta
            target_annotation.Xw = complete_annotation.Xw
            target_annotation.Yw = complete_annotation.Yw
            target_annotation.Zw = complete_annotation.Zw
            target_annotation.object_size_x = complete_annotation.object_size_x
            target_annotation.object_size_y = complete_annotation.object_size_y
            target_annotation.object_size_z = complete_annotation.object_size_z
            target_annotation.save()
            save_2d_views(target_annotation)

        person_other_worker.annotation_complete = True
        person_other_worker.save()

def sync_annotations(dataset_name):
    dataset= Dataset.objects.get(name=dataset_name)
    workers = Worker.objects.all()
    persons_complete = Person.objects.filter(dataset=dataset, annotation_complete=True)

    for person_complete in persons_complete:
        person_complete_worker = person_complete.worker
        other_workers = [worker for worker in workers if worker != person_complete_worker]
        person_complete_annotations = Annotation.objects.filter(person=person_complete)

        #print (f"Syncing {person_complete.person_id} from {person_complete.worker.workerID}...")
        print("***")
        for other_worker in other_workers:
            try:
                person_other_worker = Person.objects.get(person_id=person_complete.person_id, worker=other_worker, dataset=dataset)
                if person_other_worker.annotation_complete:
                    continue
            except Person.DoesNotExist:
                continue
            print(f"Syncing {person_complete.person_id} from {person_complete.worker.workerID} to {other_worker.workerID}...")
            for complete_annotation in person_complete_annotations:
                try:
                    target_annotation = Annotation.objects.get(person=person_other_worker, frame__frame_id=complete_annotation.frame.frame_id, frame__undistorted=complete_annotation.frame.undistorted, frame__worker=other_worker, frame__dataset=dataset)
                except Annotation.DoesNotExist:
                    target_annotation = Annotation()
                    frame,_ = MultiViewFrame.objects.get_or_create(frame_id=complete_annotation.frame.frame_id, undistorted=complete_annotation.frame.undistorted, worker=other_worker, dataset=dataset)
                    target_annotation.frame = frame
                    target_annotation.person = person_other_worker

                target_annotation.rectangle_id = complete_annotation.rectangle_id
                target_annotation.creation_method = f"sync_{person_complete_worker.workerID}__{complete_annotation.creation_method}"
                target_annotation.validated = complete_annotation.validated
                target_annotation.rotation_theta = complete_annotation.rotation_theta
                target_annotation.Xw = complete_annotation.Xw
                target_annotation.Yw = complete_annotation.Yw
                target_annotation.Zw = complete_annotation.Zw
                target_annotation.object_size_x = complete_annotation.object_size_x
                target_annotation.object_size_y = complete_annotation.object_size_y
                target_annotation.object_size_z = complete_annotation.object_size_z
                target_annotation.save()
                save_2d_views(target_annotation)
            person_other_worker.annotation_complete = True
            person_other_worker.save()

def process_tracked_location(tdet,worker,frame_id,dataset):
    world_coords = tdet["cuboidToWorldTransform"] @ np.array([0,0,0,1]).reshape(-1,1)
    world_coords = world_coords[:3,0].T
    rect_id = uuid.uuid4().__str__().split("-")[-1]
    #assumes that cuboid only rotates around z axis: ([[np.cos(theta),-np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0,0,1]
    rotation_theta = np.arccos(tdet["cuboidToWorldTransform"][0][0])

    annotation_data= {"Xw":world_coords[0],
            "Yw":world_coords[1],
            "Zw":world_coords[2],
            "object_size":tdet["objectSize"][::-1],
            "personID":tdet["trackId"],
            "rotation_theta":rotation_theta,
            "rectangleID":rect_id
            }
    
            
    frame, created = MultiViewFrame.objects.get_or_create(
        frame_id=frame_id, worker=worker,undistorted=settings.UNDISTORTED_FRAMES,dataset=dataset)
    
    person, _ = Person.objects.get_or_create(
        person_id=annotation_data['personID'],worker=worker,dataset=dataset)
    if person.annotation_complete==True:
        print("Person",person.person_id, "marked as complete. Skipping...")
        return
    # Create a new annotation object for the given person and frame
    try:
        annotation = Annotation.objects.get(
            person=person, frame=frame)
        annotation.person = person
        annotation.frame = frame
        annotation.rectangle_id = annotation_data['rectangleID']
        annotation.rotation_theta = annotation_data['rotation_theta']
        annotation.Xw = annotation_data['Xw']
        annotation.Yw = annotation_data['Yw']
        annotation.Zw = annotation_data['Zw']
        annotation.object_size_x = annotation_data['object_size'][0]
        annotation.object_size_y = annotation_data['object_size'][1]
        annotation.object_size_z = annotation_data['object_size'][2]
        annotation.creation_method = f"imported_{tdet['cam_id']}_frame_{tdet['frame_id']}"
    except Annotation.DoesNotExist:
        annotation = Annotation(
            person=person,
            frame=frame,
            rectangle_id=annotation_data['rectangleID'],
            rotation_theta=annotation_data['rotation_theta'],
            Xw=annotation_data['Xw'],
            Yw=annotation_data['Yw'],
            Zw=annotation_data['Zw'],
            object_size_x=annotation_data['object_size'][0],
            object_size_y=annotation_data['object_size'][1],
            object_size_z=annotation_data['object_size'][2],
            creation_method = f"imported_{tdet['cam_id']}_frame_{tdet['frame_id']}"
        )

    # Save the annotation object to the database
    try:
        annotation.save()
        try:
            save_2d_views(annotation)
        except Exception as e:
            print("Error saving 2d views:", e)
    except Exception as e:
        print("Error saving annotation:", e)
    
    print("Saved annotation for person", person.person_id, "in frame", frame.frame_id)

def preprocess_scout_data(frames_path: Path, calibration_path: Path, 
                          tracks_path: Path, worker_id: str, dataset_name: str, range_start: int = 0, range_end: int = 12000, testing:bool = True):
    
    # Load tracks data
    with open(tracks_path, 'rb') as f:
        tracks_data = pickle.load(f)

    if testing:
        print("Loading only 100 frames for testing...")
        range_start = 0
        range_end = 500
        new_tracks = {}
        i = 0
        for k, v in tracks_data.items():
            if np.all(v[range_start:range_end]) == 0:
                continue
            new_tracks[k] = v[range_start:range_end]
            i += 1
            # if new_tracks
            if i >= 100:
                break
        tracks_data = new_tracks
    # Create worker and dataset
    worker, _ = Worker.objects.get_or_create(workerID=worker_id)
    dataset, _ = Dataset.objects.get_or_create(name=dataset_name)
    print("Creating People...")
    people_to_create = [Person(person_id=person_id, 
                               worker=worker, 
                               dataset=dataset) for person_id in tracks_data.keys()]
    print("Bulking People...")
    Person.objects.bulk_create(people_to_create, 
                               update_conflicts=True, 
                               update_fields=['person_id', 'worker', 'dataset'], 
                               unique_fields=['person_id', 'worker', 'dataset'])
    print("Creating Frames...")
    frames_to_create = [MultiViewFrame(frame_id=frame_idx, 
            worker=worker, 
            undistorted=settings.UNDISTORTED_FRAMES, 
            dataset=dataset
            ) for frame_idx in range(range_start, range_end)]
    print("Bulking Frames...")
    MultiViewFrame.objects.bulk_create(frames_to_create, ignore_conflicts=True)
    print("Fetching Frames...")
    # Prefetch all frames once
    frames_dict = {frame.frame_id: frame for frame in 
                MultiViewFrame.objects.filter(
                    worker=worker, 
                    dataset=dataset,
                    frame_id__in=list(range(range_start, range_end))
                )}
    print("Fetching People...")
    # Get all persons at once
    people = {p.person_id: p for p in Person.objects.filter(worker=worker, dataset=dataset)}

    # Create all annotations in one go
    all_annotations = []
    for person_id, positions in tqdm(tracks_data.items(), total=len(tracks_data), desc='Creating annotations'):
        positions = np.array(positions)[range_start:range_end, :]
        if person_id not in people:
            print(f"Person {person_id} not found in database")
            continue
        person = people[person_id]
        
        all_annotations.extend([
            Annotation(
                person=person,
                frame=frames_dict[frame_idx],
                rectangle_id=uuid.uuid4().__str__().split("-")[-1],
                rotation_theta=0,
                Xw=position[0],
                Yw=position[1],
                Zw=position[2] if position.shape[0] > 2 else 0,
                object_size_x=1.7,
                object_size_y=0.6,
                object_size_z=0.6,
                creation_method="imported_scout_tracks"
            ) for frame_idx, position in enumerate(positions) if np.all(position != 0)
        ])
    print("Bulking annotations...")
    # Bulk create all annotations at once
    Annotation.objects.bulk_create(
        all_annotations,
        update_conflicts=True,
        unique_fields=['frame', 'person'],
        update_fields=['rectangle_id', 'rotation_theta', 'Xw', 'Yw', 'Zw', 
                    'object_size_x', 'object_size_y', 'object_size_z', 'creation_method']
    )
    print("Fetching Annotations...")
    # Process all 2D views at once
    all_annotations = Annotation.objects.filter(
        person__in=people.values(),
        frame__in=frames_dict.values()
    ).select_related('frame', 'person')


    save_2d_views_bulk(all_annotations)
    # for annotation in tqdm(all_annotations, total=len(all_annotations), desc='Saving 2D views'):
    #     try:
    #         save_2d_views(annotation)
    #     except Exception as e:
    #         print(f"Error saving 2D views for person {annotation.person.person_id} at frame {annotation.frame.frame_id}: {e}")

    #     cls()