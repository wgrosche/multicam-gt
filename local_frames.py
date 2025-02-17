# -*- coding: utf-8 -*-
from django.conf import settings


# Make a local copy of the dataset
# Firefox link to make local image accessible by modifying: about:config
# http://kb.mozillazine.org/Links_to_local_pages_do_not_work
import shutil
from tqdm import tqdm
from pathlib import Path
# settings.configure() /cvlabdata2/home/grosche/multicam-dev/multicam-gt/gtm_hit/static/gtm_hit/dset/SCOUT/frames/cvlabrpi1/cvlabrpi1_20240531_113100_12h00m00s000_65361_0.jpg
path_to_local_copy = Path("/cvlabdata2/home/grosche/local_frames")

FRAMENUMBER_TO_PATH = settings.FRAME_PATH_DICT

max_frame = max(FRAMENUMBER_TO_PATH.keys())

for frame_num in tqdm(range(0, max_frame + 1, 10)):
    if frame_num not in FRAMENUMBER_TO_PATH:
            continue
    
    for cam, src_path in FRAMENUMBER_TO_PATH[frame_num].items():
            # Create a subfolder for the camera if it doesn't already exist.
            cam_folder = path_to_local_copy / cam
            cam_folder.mkdir(parents=True, exist_ok=True)
            # Define the destination path with the original file name.
            src = Path(src_path)
            dst = cam_folder / src.name

            # Skip copying if the destination file already exists.
            if dst.exists():
                continue

            # Copy the file to the local subfolder.
            shutil.copy(src, dst)
    # for cam in settings.CAMS:
        # Only process if the frame number is present in the dictionary.
        
        # Iterate over each camera's frame path for the current frame number.
        