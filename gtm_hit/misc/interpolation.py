"""
Spline interpolation of annotated tracks
"""

import os
from pathlib import Path
import argparse
import json 
import re
from datetime import datetime, timedelta
from typing import Union, List
import numpy as np
from tqdm import tqdm
import cv2
import django
from django.conf import settings


from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from itertools import combinations


def static_path_to_absolute(static_path: str) -> str:
    """
    Converts a Django /static/... path to an absolute path on the filesystem.
    """
    relative = static_path.lstrip("/").replace("static/", "", 1)
    project_root = os.path.abspath(os.path.join(__file__, ".."))  # or hardcode base path
    return os.path.join(project_root, "gtm_hit", "static", relative)

def get_frame_path(frame_id, cam_id):

    frame_path = Path('/cvlabdata2/home/grosche/multicam-dev/multicam-gt/gtm_hit' + settings.FRAME_PATH_DICT.get(frame_id, {}).get(cam_id, ''))

    return frame_path

def get_frame_timestamp(frame_id, cam_id):
    frame_path = get_frame_path(frame_id=frame_id, cam_id=cam_id)
    filename = frame_path.name
    parts = filename.split('_')
    if len(parts) >= 4:
        # Parse the timestamp part (12h40m29s994)
        time_part = parts[3]
        if 'h' in time_part and 'm' in time_part and 's' in time_part:
            hours = int(time_part.split('h')[0])
            minutes = int(time_part.split('h')[1].split('m')[0])
            seconds_parts = time_part.split('m')[1].split('s')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
            
            # Convert to total seconds for comparison
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

        return total_seconds
    else:
        # print("Failed to parse filename: ", filename, "frame path: ", frame_path, f" at frame {frame_id}, for {cam_id}")
        return None
    
def get_valid_timestamp(frame_id, cameras):
    camera_timestamps = {
        camera_id: ts
        for camera_id in cameras
        if (ts := get_frame_timestamp(frame_id, camera_id)) is not None
    }


    consensus_cameras = []
    resulting_timestamp = 0
    timestamps = list(camera_timestamps.values())
    # print(timestamps)
    try:
        min_time = min(timestamps)
        max_time = max(timestamps)
    except:
        print(f"times: {timestamps}, cameras: {camera_timestamps}")
    
    # Check if all cameras are within 0.5 seconds of each other
    if max_time - min_time <= 0.5:
        consensus_cameras = list(camera_timestamps.keys())
        resulting_timestamp = np.mean(list(camera_timestamps.values()))
    else:
        # Find the largest group of cameras within 0.5 seconds of each other
        for camera_id, timestamp in camera_timestamps.items():
            group = [cam for cam, time in camera_timestamps.items() 
                        if abs(time - timestamp) <= 0.5]
            times = [time for cam, time in camera_timestamps.items() 
                        if abs(time - timestamp) <= 0.5]
            if len(group) > len(consensus_cameras):
                consensus_cameras = group
                if times:
                    resulting_timestamp = np.mean(times)
                else:
                    raise ValueError("No consensus timestamps found.")

    return consensus_cameras, resulting_timestamp


def get_valid_cameras(calibs, frame_id, ground_point, max_distance=1000):
    from .geometry import is_visible
    valid_cameras = [camera_id for camera_id in calibs if np.linalg.norm(ground_point + np.dot(calibs[camera_id].R.T, calibs[camera_id].T).flatten()) < max_distance]
    valid_cameras = [camera_id for camera_id in valid_cameras if is_visible(ground_point, camera_id)]
    return valid_cameras

def return_consensus_cams_and_time(calibs, frame_id, ground_point, max_distance=1000):
    # Extract timestamps from filenames and filter cameras based on time consensus
    # valid_cameras = get_valid_cameras(calibs, frame_id, ground_point, max_distance=max_distance)

    # if len(valid_cameras) == 0:
    #     print(f"No valid cams for frame {frame_id} at location {ground_point}, using first camera timestamp")
    #     return [], get_frame_timestamp(frame_id, settings.CAMS[0])
    
    consensus_cameras, resulting_timestamp = get_valid_timestamp(frame_id, settings.CAMS)#valid_cameras)

    
    return consensus_cameras, resulting_timestamp

def load_tracks(dataset, worker, person_ids: List[int], basepath: str = None, frame_timestamps: bool = False) -> List[dict]:
    """
    Loads tracks for multiple person_ids at once with optimized DB access.
    """
    from django.db.models import Min, Max
    from .geometry import get_projected_points, is_visible
    from ..models import Annotation, Worker, Dataset
    # Prefetch all annotations in one query
    all_annotations = Annotation.objects.filter(
        person__dataset=dataset,
        person__worker=worker,
        person__person_id__in=person_ids
    ).values(
        'person__person_id', 'Xw', 'Yw', 'Zw', 'frame__frame_id',
        'object_size_x', 'object_size_y', 'object_size_z', 'rotation_theta'
    )

    # Group annotations by person_id
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in all_annotations:
        grouped[row['person__person_id']].append(row)

    # Prefetch frame ranges
    frame_bounds = Annotation.objects.filter(
        person__dataset=dataset,
        person__worker=worker,
        person__person_id__in=person_ids
    ).values('person__person_id').annotate(
        frame_min=Min('frame__frame_id'),
        frame_max=Max('frame__frame_id')
    )
    frame_bounds_dict = {row['person__person_id']: row for row in frame_bounds}

    # Process each person_id
    tracks = []
    for person_id in tqdm(person_ids, desc="Generating Interpolated Tracks"):
        view_data = grouped.get(person_id, [])
        if not view_data:
            tracks.append(None)
            continue

        timestamps_and_coords = []
        for row in view_data:
            try:
                ground_point = [row['Xw'], row['Yw'], row['Zw']]
                _, timestamp = return_consensus_cams_and_time(settings.CALIBS, row['frame__frame_id'], ground_point)
                timestamps_and_coords.append([timestamp, *ground_point])
            except Exception as e:
                print(f"Error processing frame {row['frame__frame_id']}: {e}")
                print(ground_point)
                print(return_consensus_cams_and_time(settings.CALIBS, row['frame__frame_id'], ground_point))
                continue

        if len(timestamps_and_coords) < 2:
            tracks.append(None)
            continue

        data = np.array(timestamps_and_coords, dtype=float)
        data = data[np.argsort(data[:, 0])]
        unique_indices = np.concatenate(([True], np.diff(data[:, 0]) > 0))
        data = data[unique_indices]

        if data.shape[0] < 2:
            tracks.append(None)
            continue

        times = data[:, 0]
        coords = data[:, 1:]
        eps = 1e-10
        for i in range(1, len(times)):
            if times[i] <= times[i-1]:
                times[i] = times[i-1] + eps

        try:
            spline_x = CubicSpline(times, coords[:, 0], extrapolate = False)
            spline_y = CubicSpline(times, coords[:, 1], extrapolate = False)
            spline_z = CubicSpline(times, coords[:, 2], extrapolate = False)
            def make_interpolator(sx, sy, sz):
                return lambda t_query: np.stack([
                    sx(t_query), sy(t_query), sz(t_query)
                ], axis=-1)
            
            interpolate = make_interpolator(spline_x, spline_y, spline_z)

            bounds = frame_bounds_dict[person_id]
            sample_row = view_data[0]  # Use first row for object size + rotation
            # print("t_min", times[0], "t_max", times[-1])
            # print("world points start: ", interpolate(times[0]), " end ", interpolate(times[-1]))
            tracks.append({
                'spline_x': spline_x,
                'spline_y': spline_y,
                'spline_z': spline_z,
                'interpolate': interpolate,
                't_min': times[0],
                't_max': times[-1],
                'person_id': person_id,
                'frame_min': bounds['frame_min'],
                'frame_max': bounds['frame_max'],
                'object_size_x': sample_row['object_size_x'],
                'object_size_y': sample_row['object_size_y'],
                'object_size_z': sample_row['object_size_z'],
                'rotation': sample_row['rotation_theta']
            })
        except Exception as e:
            print(f"Error creating spline for person {person_id}: {e}")
            tracks.append(None)

    return tracks



def draw_cuboid_edges(img, projected_pts, color=(0, 255, 0), thickness=2):
    """Draws cuboid edges on an image."""
    # Define edges by indices into the projected 2D point array
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # top face
        (4, 5), (5, 7), (7, 6), (6, 4),  # bottom face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]
    for i, j in edges:
        pt1 = tuple(map(int, projected_pts[i]))
        pt2 = tuple(map(int, projected_pts[j]))
        cv2.line(img, pt1, pt2, color, thickness)
    return img


def visualize_track_on_video(track, camera, output_path="output_track.mp4", fps=10):
    from .geometry import Cuboid
    from .geometry import get_projected_points, is_visible
    frame_range = range(track['frame_min'], track['frame_max'] + 1)
    frame_shape = None
    video_writer = None

    for frame_id in tqdm(frame_range, desc=f"Rendering {camera} track"):
        img_path = static_path_to_absolute(settings.FRAME_PATH_DICT.get(frame_id, {}).get(camera, None))
        if img_path is None:
            print(f"Frame {frame_id} missing for camera {camera}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        world_point = track['interpolate'](get_frame_timestamp(frame_id, camera))
        if world_point is None or not is_visible(world_point, camera):
            continue

        print("world point: ", world_point, " timestamp ", get_frame_timestamp(frame_id, camera))
        calib = settings.CALIBS[camera]

        # --- Draw projected center point ---
        uv = get_projected_points(world_point, calib)[0]
        if uv is None:
            continue
        u, v = map(int, uv)
        img = cv2.circle(img, (u, v), radius=5, color=(0, 0, 255), thickness=-1)

        # --- Create and draw cuboid ---
        cuboid = Cuboid(calib, world_point)
        cuboid_2d = cuboid.get_cuboid_points_2d(theta=0, calib=calib)[:8]  # 8 corners only
        img = draw_cuboid_edges(img, cuboid_2d)

        # --- Initialize video writer ---
        if frame_shape is None:
            h, w = img.shape[:2]
            frame_shape = (w, h)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                frame_shape
            )

        video_writer.write(img)

    if video_writer:
        video_writer.release()
        print(f"Video saved to {output_path}")
    else:
        print("No frames were written.")


def main():
    """
    Full interpolation for a dataset for given worker. Creates a clone and interpolates
    
    """
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gtmarker.settings')
    django.setup()
    worker = Worker.objects.get(workerID = 'SCIPIO')
    dataset = Dataset.objects.get(name ='SCOUT')
    track = load_tracks(dataset, worker, person_ids = [493])
    # visualize_track_on_video(track[0], 'cvlabrpi10')
    # print('generation complete')

# if __name__ == "__main__":
#     main()

