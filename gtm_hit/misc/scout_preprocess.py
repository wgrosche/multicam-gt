import numpy as np
import uuid
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person,Dataset
from django.conf import settings
from gtm_hit.misc.db import save_2d_views_bulk
import os
from ipdb import set_trace
import os.path as osp

from pathlib import Path
from typing import Optional
import pickle
from tqdm import tqdm

def preprocess_scout_data(frames_path: Path, calibration_path: Path, 
                          tracks_path: Path, worker_id: str, dataset_name: str,
                          range_start: int = 0, range_end: int = 12000, 
                          testing:bool = True):
    
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
