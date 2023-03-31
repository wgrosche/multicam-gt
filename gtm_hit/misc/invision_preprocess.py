import cv2
import numpy as np
from IPython.display import clear_output
import time
import json
import uuid
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person, View, Annotation2DView
from django.conf import settings
from gtm_hit.misc.db import save_2d_views
import os
from ipdb import set_trace
import os.path as osp
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def preprocess_invision_data(path,worker_id):
    set_trace()
    cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    cam_id_keys = [f"cam_{cam_id[0]}_{cam_id[1]}" for cam_id in cam_id_mat]
    cam_id_keys_to_idx = dict([cam_id_keys[i],i] for i in range(len(cam_id_keys)))
    output_data = {}

    for cam_id in cam_id_mat:
        cam_id = tuple(cam_id)
        cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"
        f = open(osp.join(path,f'cam_{cam_id[0]}_{cam_id[1]}.json'))
        data = json.load(f)
        data = dict([(frame["frameId"],frame) for frame in data])
        data["id"]=cam_id
        output_data[cam_id_key] = data
        print( "Loaded json output data for camera",cam_id, "for", len(data), "frames.")

    frame_id=0
    
    while(frame_id<5000):
        print(frame_id)
        #clear_output(wait=True)
        obj_set = set()
        for i,cam_id in enumerate(cam_id_mat):
            cam_id = tuple(cam_id)
            cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"
            cam_data = output_data[cam_id_key]
            if frame_id in cam_data:
                if "tracked_locations" not in cam_data[frame_id] or cam_data[frame_id]["tracked_locations"] is None:
                    continue
                for tdet in cam_data[frame_id]["tracked_locations"]:
                    track_id = tdet["trackId"]
                    if track_id not in obj_set:
                        obj_set.add(track_id)
                        process_tracked_location(tdet,worker_id,frame_id)
        cls()
        frame_id+=1

def process_tracked_location(tdet,worker_id,frame_id):
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

    worker, _ = Worker.objects.get_or_create(workerID=worker_id)
            
    frame, created = MultiViewFrame.objects.get_or_create(
        frame_id=frame_id, worker=worker,undistorted=settings.UNDISTORTED_FRAMES)
    
    person, _ = Person.objects.get_or_create(
        person_id=annotation_data['personID'],worker=worker)
    
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
        annotation.creation_method = "imported"
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
            creation_method = "imported"
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