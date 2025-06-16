
import cv2


import os
import django


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
from interpolation import static_path_to_absolute
from PIL import Image

def load_tile(path):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), strict_map_key=False)

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def visualize_frame(frame_id, output_dir="output_frames"):
    # data = load_tile(tile_path)

    data = json.load(open("/cvlabdata2/home/grosche/multicam-dev/multicam-gt/sequence_01_testing_annotations.json"))["frames"]
    frame_data = [frame_data for frame_data in data if int(frame_data["frame_id"]) == frame_id][0]
    annotations = frame_data["annotations"]
    os.makedirs(output_dir, exist_ok=True)

    # Group annotations per camera
    cam_views = {}
    for ann in annotations:
        for cam_id, bbox in ann["projections_2d"].items():
            if cam_id not in cam_views:
                cam_views[cam_id] = []
            cam_views[cam_id].append((ann["track_id"], bbox))

    for cam_id, bboxes in cam_views.items():
        # Load image for this frame and camera
        img_path = static_path_to_absolute(settings.FRAME_PATH_DICT[frame_id][cam_id])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Draw all boxes
        for track_id, bbox in bboxes:
            color = (hash(track_id) % 256, 128, 255)  # pseudo-random color
            img = draw_bbox(img, bbox, color)
            cv2.putText(img, f"ID {track_id}", (bbox[0][0], bbox[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show result
        out_path = os.path.join(output_dir, f"frame{frame_id:04d}_{cam_id}.jpg")
        cv2.imwrite(out_path, img)
        print(f"[INFO] Saved {out_path}")


# Example usage
if __name__ == "__main__":
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gtmarker.settings')
    django.setup()
    visualize_frame(100)  # adjust path as needed