import msgpack

TILE_PATH = "tiles/frame_000000.msgpack"  # <-- Change this to the frame you want to inspect

# Load the binary .msgpack file
with open(TILE_PATH, "rb") as f:
    tile_data = msgpack.unpackb(f.read(), strict_map_key=False)

# Pretty-print the loaded data
from pprint import pprint
pprint(tile_data)