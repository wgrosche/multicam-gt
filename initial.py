"""
Run once before launching the server.
Sets up the repository as required, establishing symlinks and...

For cvlab users:

python initial.py --dset_src '/cvlabscratch/datasets/SCOUT' --dsetname SCOUT
"""








from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--dset_src', type=str, required=True)
    parser.add_argument('--dsetname', type=str, default="SCOUT")

    args = parser.parse_args()

    SYMLINK_BASE = Path(args.dset_src)

    DSETPATH = STATIC_ROOT / "gtm_hit" / "dset" / args.dsetname
    SYMLINK_DEST_FRAMES = DSETPATH / "frames"
    SYMLINK_SOURCE_FRAMES = SYMLINK_BASE / 'images' / SEQUENCE
    CALIBPATH = DSETPATH / "calibrations"
    CALIB_SRC = SYMLINK_BASE / 'calibrations'/ SEQUENCE