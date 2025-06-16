import os
import re
from datetime import timedelta
from pathlib import Path

# Configuration
# folder = Path("/cvlabdata2/home/grosche/multicam-dev/multicam-gt/gtm_hit/static/gtm_hit/dset/SCOUT/offset_frames/cvlabrpi11")
# time_offset = timedelta(seconds=2.3)
# frame_offset = 23

folder = Path("/cvlabdata2/home/grosche/multicam-dev/multicam-gt/gtm_hit/static/gtm_hit/dset/SCOUT/offset_frames/cvlabrpi22")
time_offset = timedelta(seconds=1.0)
frame_offset = 10
dry_run = False  # Set to False to actually rename files

# Regex pattern to match the timestamp part of the filename
pattern = re.compile(r"_(\d{8})_(\d{6})_(\d{2})h(\d{2})m(\d{2})s(\d{3})_(\d+)_(\d+).jpg$")

for file in folder.glob("*.jpg"):
    match = pattern.search(file.name)
    if not match:
        print(f"Skipping file with invalid format: {file.name}")
        continue

    date_str, time_str, h, m, s, ms, global_frame_id, frame_id = match.groups()

    # Convert timestamp to timedelta
    current_time = timedelta(
        hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms)
    )

    # Apply the offset
    new_time = current_time + time_offset

    # Convert back to hmsms format
    total_seconds = int(new_time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(new_time.microseconds / 1000)

    new_offset_str = f"{hours:02d}h{minutes:02d}m{seconds:02d}s{milliseconds:03d}"

    # Update frame ID
    new_frame_id = int(frame_id) + frame_offset
    new_frame_id_str = f"{new_frame_id}"  # adjust width if needed

    # Build new filename
    new_name = re.sub(
        pattern,
        f"_{date_str}_{time_str}_{new_offset_str}_{global_frame_id}_{new_frame_id_str}.jpg",
        file.name
    )

    # Show or apply rename
    print(f"{'[DRY RUN] ' if dry_run else ''}Renaming {file.name} -> {new_name}")
    if not dry_run:
        new_path = file.with_name(new_name)
        file.rename(new_path)

