import cv2
from tqdm import tqdm
import django
from django.conf import settings
import numpy as np
import os
import json
import re
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def image_hash(img):
    """Fast image hash for equality checking (MD5 over JPEG bytes)."""
    return hashlib.md5(img).hexdigest()

def process_camera_folder(cam_folder_path, cam_name, interval, offsets, regex_pattern):
    frame_paths = {}
    last_hash = None

    files = sorted(
        [f for f in os.scandir(cam_folder_path) if f.is_file() and f.name.endswith(".jpg")],
        key=lambda x: int(regex_pattern.search(x.name).group(1))
    )

    for file in files:
        match = regex_pattern.search(file.name)
        if not match:
            continue
        index = int(match.group(1))
        offset = offsets.get(cam_name, 0)

        root_stub = '/static' + cam_folder_path.split("/static")[1]
        frame_paths[index] = os.path.join(root_stub, file.name)

    return cam_name, frame_paths


def get_frame_path_dict(
    frame_path=settings.SYMLINK_DEST_FRAMES,
    local_path=None,
    cache_path="utility/interpolate_cache.json",
    interval=10,
    force_reload: bool = True,
    timestamped:bool = False
):
    from .interpolation import get_frame_timestamp
    if not force_reload and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return {int(k): v for k, v in json.load(f).items()}

    lookup_path = local_path if local_path is not None else frame_path
    offsets = settings.OFFSETS
    regex_pattern = re.compile(r"_(\d+)\.jpg$")
    frame_path_dict = {}

    with ThreadPoolExecutor() as executor:
        futures = []
        for cam_folder in os.scandir(lookup_path):
            if not cam_folder.is_dir():
                continue
            cam_name = cam_folder.name

            folder_path = cam_folder.path
            if cam_name in offsets:
                if "frames" in folder_path:
                    folder_path = folder_path.replace("frames", "offset_frames")

            futures.append(executor.submit(
                process_camera_folder, folder_path, cam_name, interval, offsets, regex_pattern
            ))

        for future in tqdm(futures, desc="Processing cameras"):
            cam_name, frames = future.result()
            for index, path in frames.items():
                if index not in frame_path_dict:
                    frame_path_dict[index] = {}
                
                if timestamped:
                    timestamp = get_frame_timestamp(index, cam_name)
                    frame_path_dict[index][cam_name] = {"path": path, "time_s" : timestamp}
                else:
                    frame_path_dict[index][cam_name] = path

    with open(cache_path, 'w') as f:
        json.dump(frame_path_dict, f)

    return frame_path_dict

if __name__ == "__main__":
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gtmarker.settings')
    django.setup()

    FRAME_PATH_DICT = get_frame_path_dict(
        dset = settings.DSETNAME, 
        frame_path = settings.SYMLINK_DEST_FRAMES, 
        cams = settings.CAMS, 
        interval = 1,
        cache_path = 'timestamped_paths.json'
        )