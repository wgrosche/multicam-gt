import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gtmarker.settings')
django.setup()

from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person,Dataset, Annotation2DView, View
import argparse
import json
import numpy as np
from datetime import datetime
from django.db.models import Min, Max


from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person, Dataset, Annotation2DView, View
import argparse
import json 
import re
from datetime import datetime, timedelta
from typing import Tuple

from PIL import Image




def generate_projections_for_frame_and_track(track, frame_id, timestamp) -> dict:
    from interpolation import get_frame_timestamp, get_valid_cameras
    from gtm_hit.misc.geometry import Cuboid
    interpolator = track['interpolate']
    world_coord = interpolator(timestamp)
    valid_cams = get_valid_cameras(settings.CALIBS, frame_id, world_coord)
    return_dict = {}
    
    for camera in valid_cams:
        frame_timestamp = get_frame_timestamp(frame_id, camera)
        frame_world_coord = interpolator(frame_timestamp)
        if np.any(np.isnan(np.array(frame_world_coord))):
            continue
        cuboid = Cuboid(
                    calib = settings.CALIBS[camera], 
                    world_point=frame_world_coord, 
                    width = track['object_size_y'], 
                    length = track['object_size_z'], 
                    height = track['object_size_x'])
        return_dict[camera] = cuboid.get_bbox()

    return return_dict
    

def generate_frame_annotation_for_track(
          track:dict, 
          frame_id:int,
          timestamp_global = None
          ) -> dict:
    # from interpolation import get_valid_timestamp
    interpolator = track['interpolate']
    # if timestamp_global is None:
    #     _, timestamp_global = get_valid_timestamp(frame_id, settings.CAMS)
    # if timestamp_global < track['t_min'] or timestamp_global > track['t_max']:
    #     return None
    # print(timestamp_global)
    world_coords = interpolator(timestamp_global)
    if np.isnan(world_coords).any():
        return None
    frame_annotation = {
         "track_id": track['person_id'],
         "cuboid_3d": {
            "Xw": world_coords[0], 
            "Yw": world_coords[1], 
            "Zw": world_coords[2],
            "width": track['object_size_x'], #x
            "length" : track['object_size_y'], #y
            "height" : track['object_size_z'], #z
            "rotation_theta": track['rotation']
            },
         "projections_2d": generate_projections_for_frame_and_track(track, frame_id, timestamp_global)
        }
    
    return frame_annotation

# def generate_frame_annotations_for_track(
#           track:dict
#           ) -> dict:
#     from interpolation import get_valid_timestamp, get_valid_cameras
#     from gtm_hit.misc.geometry import Cuboid

#     frame_annotations = {frame_id: {
#          "track_id": track['person_id'],
#          "timestamp" : ts,
#          "cuboid_3d": {
#             "Xw": world_coords[0], 
#             "Yw": world_coords[1], 
#             "Zw": world_coords[2],
#             "width": track['object_size_x'], 
#             "length" : track['object_size_y'], 
#             "height" : track['object_size_z'],
#             "rotation_theta": track['rotation']
#             },
#          "projections_2d": {
#             camera: Cuboid(
#                 calib = settings.CALIBS[camera], 
#                 world_point=world_coords, 
#                 width = track['object_size_x'], 
#                 length = track['object_size_y'], 
#                 height = track['object_size_z']).get_bbox() for camera in get_valid_cameras(settings.CALIBS, None, world_coords)
#                 }
#             } for frame_id in range(track['frame_min'], track['frame_max']) if (ts := get_valid_timestamp(frame_id, settings.CAMS)[1]) is not None and not np.isnan(world_coords := track['interpolate'](ts)).any()}
    
#     return frame_annotations

def as_martin(dataset, worker, sequence:str = 'sequence_01', testing:bool = False):
    """
    Returns dict of annotations:
    2d projections are only shown if visible
    ```
        {
        “sequence_id”: “seq01",
        “total_frames”: 12000, // Assuming 20 mins @ 10 FPS for example
        “frames”: [
            {
            “frame_id”: 0,
            “timestamp_ms”: 0,
            “annotations”: [
                {
                “track_id”: 1,
                        “cuboid_3d”: { // Parameters defining the 3D cuboid
                            “center_x”: 1.2, “center_y”: 0.5, “center_z”: 2.0,
                            “size_x”: 0.8, “size_y”: 0.6, “size_z”: 1.5,
                            “rotation_yaw”: 0.78 // radians
                            // Add other necessary parameters like rotation_pitch, rotation_roll if applicable
                        },
                        “projections_2d”: {
                            "cvlabrpi1": [100, 150, 50, 75], # x1, y1, x2, y2
                            "cvlabrpi1": [200, 180, 60, 80]
                            // ... for all 26 cameras
                        }
                        },
                        // ... other tracks in this frame
                    ]
                    },
                    // ... other frames
                ]
                }
    ```
    Not everything is needed for cuboid the center and height should be enough, for bbox visible flag is not needed only include camera is it’s visible
    For performance it might also be useful to have a a json for trajectory so we can look up trajectory instantly instead of having to go through all the annoation something like:
    {
                “sequence_id”: “seq01”,
                “trajectories”: [
                    {
                    “track_id”: 1,
                    “points_3d”: [ // Array of [x, y, z, frame_id]
                        [1.2, 0.5, 2.0, 0],
                        [1.22, 0.51, 2.0, 1],
                        // ...
                    ]
                    },
                    // ... other tracks
                ]
                }
    """
    from interpolation import load_tracks, get_valid_timestamp

    
    if testing:
        frames = MultiViewFrame.objects.filter(worker=worker, dataset=dataset).order_by('frame_id').values('frame_id')[:300]
    else:
        frames = MultiViewFrame.objects.filter(worker=worker, dataset=dataset).order_by('frame_id').values('frame_id')

    # frame_ids = frames.values_list('frame_id', flat=True)
    # max_frame = max([frame['frame_id'] for frame in frames])
    # min_frame = min([frame['frame_id'] for frame in frames])
    # Get min and max frame IDs in one DB query
    frame_range = frames.aggregate(
        min_frame_id=Min('frame_id'),
        max_frame_id=Max('frame_id')
    )
    max_frame = frame_range['max_frame_id']#max([frame['frame_id'] for frame in frames])
    min_frame = frame_range['min_frame_id']#min([frame['frame_id'] for frame in frames])

    # Use Python range for frame IDs
    frame_ids = range(min_frame, max_frame + 1)

    # filter out if only ~3 annotations
    people = Person.objects.filter(worker= worker, dataset = dataset, annotation__frame__frame_id__in=frame_ids).order_by('person_id').distinct().values('person_id')

    

    # Preload valid tracks once
    people_ids = [p["person_id"] for p in people]
    tracks = load_tracks(dataset, worker, person_ids=people_ids)  # Batched version

    valid_tracks = {
        track['person_id']: track
        for track in tracks
        if track is not None
    }

    for id, track in valid_tracks.items():
        print(f"track {id}: start frame: {track['frame_min']} end frame: {track['frame_max']}")

    from tqdm import tqdm

    def make_frame(frame_id):
        ts = get_valid_timestamp(frame_id, settings.CAMS)[1]
        annotations = []
        for pid, track in valid_tracks.items():
            if track['frame_min'] <= frame_id <= track['frame_max']:
                annotations.append(generate_frame_annotation_for_track(track, frame_id, timestamp_global = ts))
        return {
            "frame_id": frame_id, #TODO adjust output frame id by offset
            "timestamp_ms": ts * 1000,
            "annotations": [annotation for annotation in annotations if annotation is not None]
        }

    frames = list(tqdm(map(make_frame, frame_ids), total=max_frame - min_frame, desc="Compiling frames"))


    return_dict = {
        "sequence_id": sequence,
        "total_frames": int(max_frame - min_frame) + 1,
        "frames": frames
    }
    print("Saving Annotation Dictionary")
    if testing:
        sequence = f"{sequence}_testing"
    filepath = f"{sequence}_annotations.json"
    with open(filepath, 'w') as f:
            json.dump(return_dict, f)
        


def visualise_export(dictionary:dict):
    pass




"""
single frame file format
frame_id.json = {cam_name:{id:bbox, ...}, ..., "world_coord":{id:3dpoint, ...}}


"""
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SCOUT')
    parser.add_argument('--worker', type=str, default='SCIPIO')
    parser.add_argument('--output', type=str, default='coco')
    # parser.add_argument('--output', type=str, required=True, help='Output file path')

    args = parser.parse_args()
    print('Exporting data for dataset: {} and worker: {}'.format(args.dataset, args.worker))
    dataset = Dataset.objects.get(name = args.dataset)
    worker = Worker.objects.get(workerID = args.worker)
    as_martin(dataset, worker, sequence = 'sequence_01', testing=False)

    # frames = MultiViewFrame.objects.filter(dataset__name=args.dataset, worker__workerID=args.worker, frame_id__in=range(0, 2000))
    # annotations = Annotation.objects.filter(person__dataset__name=args.dataset, person__worker__workerID=args.worker, frame__in=frames)


    # export_dict = {}
    # for view in View.objects.all():
    #     print('Exporting data for view: {}'.format(view.view_id))
    #     view_data = Annotation2DView.objects.filter(annotation__in=annotations, view=view).values('annotation__person__person_id', 'x1', 'y1', 'x2', 'y2', 'annotation__Xw', 'annotation__Yw', 'annotation__Zw', 'annotation__frame__frame_id')
    #     for view_data_entry in view_data:
    #         view_id = view.view_id
    #         person_id = view_data_entry['annotation__person__person_id']
    #         x1, y1, x2, y2 = view_data_entry['x1'], view_data_entry['y1'], view_data_entry['x2'], view_data_entry['y2']
    #         xw, yw, zw = view_data_entry['annotation__Xw'], view_data_entry['annotation__Yw'], view_data_entry['annotation__Zw']
    #         frame_id = view_data_entry['annotation__frame__frame_id']
    #         if [x1, x2, y1, y2] == [-1, -1, -1, -1]:
    #             continue
    #         camera = settings.CAMS[view_id]
    #         if camera not in export_dict:
    #             export_dict[camera] = {}
    #         if frame_id not in export_dict[camera]:
    #             export_dict[camera][frame_id] = []
    #         export_dict[camera][frame_id].append({
    #             'person_id': person_id,
    #             'bbox': [x1, y1, x2, y2],
    #             'pos': [xw, yw, zw]
    #         })
    #     timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #     filepath = f'exports/{args.dataset}_{args.worker}_{settings.CAMS[view_id]}_{timestamp}.json'
    #     with open(filepath, 'w') as f:
    #         json.dump(export_dict[camera], f)

if __name__ == '__main__':
    main()
