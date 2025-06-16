
"""
How to use

import msgpack from "@ygoe/msgpack";

async function loadFrame(frameId) {
  const padded = frameId.toString().padStart(4, '0');
  const res = await fetch(`/tiles/frame_${padded}.msgpack`);
  const buf = await res.arrayBuffer();
  const frameData = msgpack.decode(new Uint8Array(buf));
  return frameData;
}
"""


import os
import json
import msgpack
from tqdm import tqdm

INPUT_JSON = "/cvlabdata2/home/grosche/multicam-dev/multicam-gt/sequence_01_testing_annotations.json"
OUTPUT_DIR = "tiles"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def flatten_projection(proj):
    """Convert [[x1, y1], [x2, y2]] to [x1, x2, y1, y2]"""
    return [proj[0][0], proj[1][0], proj[0][1], proj[1][1]]

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

for frame in tqdm(data["frames"], desc="Exporting tiles"):
    frame_id = frame["frame_id"]
    timestamp = frame["timestamp_ms"]

    annotations = []
    for ann in frame["annotations"]:
        cuboid = ann["cuboid_3d"]
        projections = {
            cam_id: flatten_projection(bbox)
            for cam_id, bbox in ann["projections_2d"].items()
        }

        annotation = {
            "track_id": ann["track_id"],
            "cuboid_3d": (
                cuboid["Xw"], cuboid["Yw"], cuboid["Zw"],
                cuboid["width"], cuboid["length"], cuboid["height"],
                cuboid["rotation_theta"]
            ),
            "projections_2d": projections
        }
        annotations.append(annotation)

    tile_data = {
        "frame_id": frame_id,
        "timestamp_ms": timestamp,
        "annotations": annotations
    }

    tile_path = os.path.join(OUTPUT_DIR, f"frame_{frame_id:06d}.msgpack")
    with open(tile_path, "wb") as f:
        f.write(msgpack.packb(tile_data))