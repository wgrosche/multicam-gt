"""
Spline interpolation of annotated tracks
"""

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
from datetime import datetime


from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person, Dataset, Annotation2DView, View
import argparse
import json 
import re
from datetime import datetime, timedelta

from PIL import Image

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Union, List
import os

def get_time(frame_id: int, basepath: str) -> datetime:
    # Find the file matching the pattern
    filename = next(Path(basepath).glob(f"*_{frame_id}.jpg"), None)
    
    if filename is None:
        raise FileNotFoundError(f"No file found for frame_id {frame_id} in {basepath}")
    
    # Convert Path object to string for regex
    filename_str = str(filename)
    
    # Extract start datetime and image time offset
    match = re.search(r"_(\d{8})_(\d{6})_(\d{2})h(\d{2})m(\d{2})s(\d{3})_", filename_str)
    
    if match:
        date_str, time_str, h, m, s, ms = match.groups()
        start_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        
        # Parse current offset time
        current_time = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
        
        # Reference time is 12:00:00.000
        ref_time = timedelta(hours=12)
        
        # Actual offset = current_time - ref_time
        offset = current_time - ref_time
        
        # Final image timestamp
        image_dt = start_dt + offset
        return image_dt
    else:
        raise ValueError(f"Filename format invalid: {filename_str}")


from typing import Union, List
import numpy as np
from scipy.interpolate import CubicSpline, BSpline
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from tqdm import tqdm

def load_track(dataset, worker, person_id: Union[int, List[int]] = None, basepath: str = None, frame_timestamps:bool=False) -> List[dict]:
    """
    frame_timestamps(bool): if true will evaluate the spline at timestamps corresponding to the frame id.
    """
    if isinstance(person_id, list):
        person_ids = person_id
    else:
        person_ids = [person_id]
    
    tracks = []
    
    # Fetch raw annotations
    for person_id in tqdm(person_ids, desc="Generating Interpolated Tracks"):
        view_data = Annotation.objects.filter(
            person__dataset=dataset,
            person__worker=worker,
            person__person_id=person_id
        ).values('Xw', 'Yw', 'Zw', 'frame__frame_id', 'frame__timestamp')
        
        if not view_data:
            tracks.append(None)
            continue
        
        # Extract timestamps and coordinates
        timestamps_and_coords = []
        for row in view_data:
            try:
                timestamp = get_time(row['frame__frame_id'], basepath)  # This returns a datetime object
                # Convert datetime to float (seconds since epoch) for interpolation
                timestamp_float = mdates.date2num(timestamp)
                timestamps_and_coords.append([timestamp_float, row['Xw'], row['Yw'], row['Zw']])
            except Exception as e:
                print(f"Error processing frame {row['frame__frame_id']}: {e}")
                continue
        
        if not timestamps_and_coords:
            tracks.append(None)
            continue
            
        data = np.array(timestamps_and_coords, dtype=float)
        
        if data.shape[0] < 2:
            tracks.append(None)
            continue
        
        # Sort by time
        data = data[np.argsort(data[:, 0])]
        
        # Remove duplicate timestamps by keeping only the first occurrence
        unique_indices = np.concatenate(([True], np.diff(data[:, 0]) > 0))
        data = data[unique_indices]
        
        # Check if we still have enough data points
        if data.shape[0] < 2:
            tracks.append(None)
            continue
        
        times = data[:, 0]  # Shape (N,)
        coords = data[:, 1:]  # Shape (N, 3)
        
        # Add a small epsilon to any timestamps that are still identical
        # This is a fallback in case the unique_indices approach doesn't catch all duplicates
        eps = 1e-10
        for i in range(1, len(times)):
            if times[i] <= times[i-1]:
                times[i] = times[i-1] + eps
        
        try:
            # Create splines
            spline_x = CubicSpline(times, coords[:, 0])
            spline_y = CubicSpline(times, coords[:, 1])
            spline_z = CubicSpline(times, coords[:, 2])
            
            # Unified interpolation function
            def interpolate(t_query):
                return np.stack([
                    spline_x(t_query),
                    spline_y(t_query),
                    spline_z(t_query)
                ], axis=-1)
            
            # Generate interpolated track at 1-second intervals
            t_min = times[0]
            t_max = times[-1]
            # Create evenly spaced points (1 second intervals in matplotlib date format)
            # 1 second = 1/(24*60*60) in days
            # t_interp = np.arange(t_min, t_max + 1/(24*60*60), 1/(24*60*60))
            
            n_seconds = int((t_max - t_min) * 24 * 60 * 60) + 1
            t_interp = np.linspace(t_min, t_max, n_seconds)
            interpolated_track = interpolate(t_interp)  # Shape (M, 3)
            # Convert interpolated times back to datetime for storage
            interp_datetimes = [mdates.num2date(t) for t in t_interp]
            
            tracks.append({
                'spline_x': spline_x,
                'spline_y': spline_y,
                'spline_z': spline_z,
                'interpolate': interpolate,
                't_min': t_min,
                't_max': t_max,
                'interpolated_track': interpolated_track,  # (M, 3)
                'interpolated_times': t_interp,  # (M,) as float
                'interpolated_datetimes': interp_datetimes  # (M,) as datetime objects
            })
        except Exception as e:
            print(f"Error creating spline for person {person_id}: {e}")
            tracks.append(None)
    
    return tracks

import matplotlib.pyplot as plt

def visualize_track(tracks:List[dict], person_id:Union[int, List[int]]=None) -> None:
    """
    Visualizes the raw and interpolated X-Y world coordinates.
    
    Parameters:
    - track: dict returned by `load_track(...)`
    - person_id: optional, for title labeling
    """
    plt.figure(figsize=(8, 6))
    for track, person_id in zip(tracks, person_id):
        if track is not None:
            # Raw data (sampled)
            raw_x = track['spline_x'].x  # times
            raw_y = track['spline_y'](raw_x)

            # Interpolated data
            interp_xy = track['interpolated_track'][:, :2]

            
            plt.plot(interp_xy[:, 0], interp_xy[:, 1], label="Interpolated Track", color='blue')
            plt.scatter(track['spline_x'](raw_x), raw_y, label="Original Data", color='red', marker='x')

    plt.xlabel("X (World Coord)")
    plt.ylabel("Y (World Coord)")
    # title = f"2D World Track (Person {person_id})" if person_id else "2D World Track"
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # Keep aspect ratio equal for world coords
    plt.tight_layout()
    plt.savefig('test_track_interpolate.jpg', dpi=500)


worker = Worker.objects.get(workerID = 'SCIPIO')
dataset = Dataset.objects.get(name ='SCOUT')
track = load_track(dataset, worker, [i for i in range(5)], basepath = "/cvlabdata2/home/grosche/multicam-dev/multicam-gt/gtm_hit/static/gtm_hit/dset/SCOUT/frames/cvlabrpi1/")

visualize_track(track, person_id=[i for i in range(5)])

def main():
    """
    Full interpolation for a dataset for given worker. Creates a clone and interpolates
    
    """