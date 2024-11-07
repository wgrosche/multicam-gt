import numpy as np
import uuid
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person,Dataset
from django.conf import settings
from gtm_hit.misc.db import save_2d_views_bulk, save_2d_views
from pathlib import Path
import pickle
from tqdm import tqdm
import time
import os
import h5py
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from collections import namedtuple
from dataclasses import dataclass
from .geometry import project_2d_points_to_mesh, reproject_to_world_ground_batched

Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])
from trimesh.base import Trimesh


def preprocess_scout_data(tracks_path: Path, 
                          worker_id: str, 
                          dataset_name: str,
                          range_start: int = 0, 
                          range_end: int = 12000, 
                          testing:bool = True
                          ):
    
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

    # for annotation in tqdm(all_annotations, total=len(all_annotations), desc='Saving 2D views'):
    #     print(annotation)
    #     save_2d_views(annotation)
    save_2d_views_bulk(all_annotations)


class HDF5FrameStore:
    def __init__(self, file_name):
        self.file_name = file_name + '.h5'
        # self.initialize_hdf5()

    # Initialize or open the HDF5 file in append mode
    def initialize_hdf5(self):
        #create parent directory if it does not exist
        parent_dir = os.path.dirname(self.file_name)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        with h5py.File(self.file_name, 'a') as h5f:
            # No need to create anything at initialization, just ensuring the file exists
            pass

    # Save a single frame to HDF5
    def save_frame(self, group_name, frame_data):
        with h5py.File(self.file_name, 'a') as h5f:
            if group_name in h5f:
                del h5f[group_name]  # Overwrite if frame already exists
            group = h5f.create_group(group_name)
            group.create_dataset('instances_id', data=frame_data['instances_id'])
            group.create_dataset('labels', data=frame_data['labels'])
            group.create_dataset('scores', data=frame_data['scores'])
            group.create_dataset('bboxes', data=frame_data['bboxes'])

    # Load a specific frame from HDF5
    def load_frame(self, frame_name):
        with h5py.File(self.file_name, 'r') as h5f:
            group = h5f[frame_name]
            return {
                "instances_id": np.array(group['instances_id']),
                "labels": np.array(group['labels']),
                "scores": np.array(group['scores']),
                "bboxes": np.array(group['bboxes'])
            }
        
    def load_frames(self, frame_names: list):
        frames = {}
        with h5py.File(self.file_name, 'r') as h5f:
            for frame_name in frame_names:
                group = h5f[frame_name]
                frames[frame_name] = {
                    "instances_id": group['instances_id'][()],
                    "labels": group['labels'][()],
                    "scores": group['scores'][()],
                    "bboxes": group['bboxes'][()]
                }
        return frames
    
        # Optional: Load all frames if needed
    def load_all_frames(self):
        frames = {}
        with h5py.File(self.file_name, 'r') as h5f:
            for frame_name in h5f:
                frames[frame_name] = self.load_frame(frame_name)

        return frames

    def get_all_frame_names(self):
        with h5py.File(self.file_name, 'r') as h5f:
            names = [frame_name for frame_name in h5f]

        return names
    

    
def load_trajectories_3d_as_dict(
        sequence:str, 
        camera:str, 
        calibration:Dict[str, Calibration], 
        mesh:Trimesh, 
        hdf5_template:Optional[str] = "/cvlabdata2/home/grosche/dev/calibration/sync_frame_seq_1/{camera}",
        frame_step:int = 1,
        ) -> Dict[str, List[Tuple[np.ndarray, int, int, int]]]:

    # Load predictions from HDF5
    preds = HDF5FrameStore(hdf5_template.format(frame_seq=sequence, camera=camera))
    # skip frames according to frame_step
    frames = preds.get_all_frame_names()
    frames = frames[::frame_step]
    
    start_time = time.time()
    predicted_frames = preds.load_frames(frames)
    load_frames_time = time.time() - start_time
    
    print(f"Load frames time: {load_frames_time:.2f}s for {len(frames)} frames")

    num_frames = len(frames)
    calib = calibration[camera]

    # Dictionary to store trajectories
    trajectories = {}
    # Initialize tqdm progress bar
    pbar = tqdm(total=num_frames, desc=f"Processing {camera}")
    
    total_loop_time = 0
    total_project_time = 0

    # Process all frames
    for frame_id, frame in enumerate(frames):
        loop_start_time = time.time()
        
        pred = predicted_frames[frame]
        instance_ids = pred['instances_id']
        bboxes = pred['bboxes']

        if len(instance_ids) == 0:
            pbar.update(1)
            continue

        # Calculate bottom centers in 2D for all detections in the frame
        bottom_centers_2d = np.array([
            [(x_min + x_max) / 2, y_max] for x_min, y_min, x_max, y_max in bboxes
        ])

        # Project 2D points to 3D ground points
        project_start_time = time.time()
        if settings.FLAT_GROUND:
            K0, R0, T0 = calib.K, calib.R, calib.T
            ground_points = reproject_to_world_ground_batched(bottom_centers_2d, K0, R0, T0)
        else:
            ground_points = project_2d_points_to_mesh(bottom_centers_2d, calib, mesh)
        project_time = time.time() - project_start_time
        total_project_time += project_time

        # Update trajectories
        for idx, instance_id in enumerate(instance_ids):
            point = np.array(ground_points[idx]).astype(float)
            # print(type(point))
            if np.any(np.isfinite(point)):
                if instance_id not in trajectories:
                    trajectories[instance_id] = ([], frame_id, frame_id, instance_id)
                traj, start, end, _ = trajectories[instance_id]
                last_frame = end
                if frame_id > last_frame + 1:
                    # Insert NaNs for the missed frames
                    num_missed_frames = frame_id - last_frame - 1
                    last_point = traj[-1] if traj else point  # Use last known point for interpolation
                    
                    # Linear interpolation for missed frames
                    for i in range(1, num_missed_frames + 1):
                        interp_frame = last_frame + i
                        interp_point = [
                            np.interp(interp_frame, [last_frame, frame_id], [last_point[j], point[j]]) for j in range(3)
                        ]
                        traj.append(interp_point)

                traj.append(point)

                trajectories[instance_id] = (traj, min(start, frame_id), max(end, frame_id), instance_id)


        loop_time = time.time() - loop_start_time
        total_loop_time += loop_time
        
        # Update progress bar
        avg_loop_time = total_loop_time / (frame_id + 1)
        avg_project_time = total_project_time / (frame_id + 1)
        pbar.set_description(f"Processing {camera} | Avg loop: {avg_loop_time:.4f}s | Avg project: {avg_project_time:.4f}s | Last loop: {loop_time:.4f}s")
        pbar.update(1)

    pbar.close()

    # Convert trajectories to the required format
    result = []
    for instance_id, (traj, start, end, _) in trajectories.items():
        result.append((np.array(traj), start, end, instance_id))
    
    total_time = time.time() - start_time
    print(f"Total time for {camera}: {total_time:.2f}s | Load frames time: {load_frames_time:.2f}s")

    return result

def preprocess_scout_data_from_dict(hdf5_template:str,
                          worker_id: str, 
                          dataset_name: str,
                          fps: int,
                          ):

    frame_step = int(10 // fps)
    print(frame_step)
    print(type(frame_step))
    range_start = int(settings.FRAME_START)
    range_end = int(settings.FRAME_END)
    traj_dict_3d = {}
    for camera in settings.CAMS:
        print(f"Processing {camera}...")
        traj_dict_3d[camera] = (load_trajectories_3d_as_dict('sync_frame_seq_1', camera, settings.CALIBS, settings.MESH, hdf5_template=hdf5_template, frame_step = frame_step))

    all_tracks_3d = {}
    global_to_local = []
    current_idx = 0
    for cam, trajs in traj_dict_3d.items():
        for local_idx, traj in enumerate(trajs):
            all_tracks_3d[current_idx] = traj
            global_to_local.append((cam, local_idx))
            current_idx += 1

    
    # Create worker and dataset
    worker, _ = Worker.objects.get_or_create(workerID=worker_id)
    dataset, _ = Dataset.objects.get_or_create(name=dataset_name)

    print("Creating People...")
    people_to_create = [Person(person_id=person_id, 
                               worker=worker, 
                               dataset=dataset) for person_id in all_tracks_3d.keys()]
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
            ) for frame_idx in range(range_start, range_end, frame_step)]
    print("Bulking Frames...")
    MultiViewFrame.objects.bulk_create(frames_to_create, ignore_conflicts=True)
    print("Fetching Frames...")
    # Prefetch all frames once
    frames_dict = {frame.frame_id: frame for frame in 
                MultiViewFrame.objects.filter(
                    worker=worker, 
                    dataset=dataset,
                    frame_id__in=list(range(range_start, range_end, frame_step))
                )}
    print("Fetching People...")
    # Get all persons at once
    people = {p.person_id: p for p in Person.objects.filter(worker=worker, dataset=dataset)}

    # Create all annotations in one go
    all_annotations = []
    for person_id, tracks in tqdm(all_tracks_3d.items(), total=len(all_tracks_3d), desc='Creating annotations'):
        positions, start, end, id = tracks

        if person_id not in people:
            print(f"Person {person_id} not found in database")
            continue

        person = people[person_id]
        
        all_annotations.extend([
            Annotation(
                person=person,
                frame=frames_dict[start + frame_idx],
                rectangle_id=uuid.uuid4().__str__().split("-")[-1],
                rotation_theta=0,
                Xw=position[0],
                Yw=position[1],
                Zw=position[2] if position.shape[0] > 2 else 0,
                object_size_x=1.7,
                object_size_y=0.6,
                object_size_z=0.6,
                creation_method="imported_scout_tracks"
            ) for frame_idx, position in enumerate(positions) if start + frame_idx in frames_dict
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

    # for annotation in tqdm(all_annotations, total=len(all_annotations), desc='Saving 2D views'):
    #     print(annotation)
    #     save_2d_views(annotation)
    save_2d_views_bulk(all_annotations)