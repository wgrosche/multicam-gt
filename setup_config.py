from pathlib import Path
import shutil
from dataclasses import dataclass
import logging
from pathlib import Path
# from django.conf import settings

# Configure logging
log_path = Path("setup_logs")
log_path.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path / 'setup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



# logger.warning("Missing configuration file, using defaults...")
# logger.error("Failed to create database connection")


@dataclass
class SetupArgs:
    DSETNAME = "WILKEANNOTATE"
    SYMLINK_SOURCE_FRAMES = Path('/cvlabscratch/home/engilber/datasets/SCOUT/collect_30_05_2024/sync_frame_seq_1')
    CALIB_SRC = Path("/cvlabscratch/home/engilber/datasets/SCOUT/collect_30_05_2024/sync_frame_seq_1/calibrations/calibrations")
    TRACKS_SRC = Path("/cvlabdata2/home/grosche/dev/calibration/merged_tracks.pkl")
    MESH_SRC = None #Path("/cvlabdata2/home/grosche/dev/calibration/merged_mesh.pkl")


def setup_dataset(DSETNAME):
    DSETPATH = Path("./gtm_hit/static/gtm_hit/dset/") / DSETNAME
    if not DSETPATH.exists():
        DSETPATH.mkdir(parents=True)
    return DSETPATH

def symlink_frames(SYMLINK_SOURCE_FRAMES, DSETPATH):
    SYMLINK_DEST_FRAMES = DSETPATH / "frames"
    if not SYMLINK_DEST_FRAMES.exists():
        SYMLINK_DEST_FRAMES.symlink_to(SYMLINK_SOURCE_FRAMES)


def copy_calibrations(CALIB_SRC, DSETPATH):
    CALIBPATH = DSETPATH / "calibrations"
    if not CALIBPATH.exists():
        CALIBPATH.mkdir(parents=True)

    for file in CALIB_SRC.glob('*.json'):
        shutil.copy2(file, CALIBPATH)

def load_tracks(PICKLEPATH, DSETPATH):
    assert PICKLEPATH.exists(), 'No pickled tracks data found'
    TRACKDIR = DSETPATH / 'tracks'
    if not TRACKDIR.exists():
        TRACKDIR.mkdir()
    shutil.copy2(PICKLEPATH, TRACKDIR)

def undistort_frames(DSETPATH):
    from gtm_hit.misc.geometry import undistort_frames
    undistort_frames(DSETPATH)


def load_mesh(MESH_SRC, DSETPATH):
    MESHDIR = DSETPATH / 'mesh'
    if not MESHDIR.exists():
        MESHDIR.mkdir()
    shutil.copy2(MESH_SRC, MESHDIR)

args = SetupArgs()

def main():
    logger.info("Starting database setup...")
    DSETPATH = setup_dataset(args.DSETNAME)
    logger.info("Creating symlinks...")
    symlink_frames(args.SYMLINK_SOURCE_FRAMES, DSETPATH)
    logger.info("Copying calibration files...")
    copy_calibrations(args.CALIB_SRC, DSETPATH)
    load_tracks(args.TRACKS_SRC, DSETPATH)

    # undistort_frames(DSETPATH)
    
    
    

if __name__ == "__main__":
    main()







