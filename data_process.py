# from gtm_hit.misc.invision_preprocess import preprocess_invision_data

from gtm_hit.misc.scout_preprocess import preprocess_scout_data
# import argparse
# parser = argparse.ArgumentParser(description="Save invision data to the annotation tool DB.")
# parser.add_argument('-i', "--input_path", default="gtm_hit/labels/json_output", help="Input path to the invision JSON data.")
# parser.add_argument('-w',"--worker_id",default="INVISION1", help="Worker ID (default: INVISION1).")
# parser.add_argument('-d',"--dataset_name", default="invision", help="Dataset name. It should match the name of the folder in dset/ (default: invision).")
# args = parser.parse_args()

#Note: undistortion and increment is determined by the settings file.
class Args:
    def __init__(self,
                 frames_path="",
                 calibration_path="",
                 tracks_path="",
                 input_path="gtm_hit/static/gtm_hit/labels/json_output",
                 worker_id="SYNC17APR0908",
                 dataset_name="13apr", 
                 range_start=3150,
                 range_end=5000):
        
        self.frames_path=frames_path,
        self.calibration_path=calibration_path,
        self.tracks_path=tracks_path,
        # self.input_path = input_path
        self.worker_id = worker_id
        self.dataset_name = dataset_name
        self.range_start=range_start
        self.range_end=range_end

args = Args()

preprocess_scout_data(args.input_path, worker_id=args.worker_id,dataset_name=args.dataset_name,range_start=args.range_start,range_end=args.range_end)

from pathlib import Path

frames_path = Path("cams")
calibration_path = Path("cam_calib")
tracks_path = Path("tracks.pkl")

preprocess_scout_data(
    frames_path=args.frames_path,
    calibration_path=args.calibration_path,
    tracks_path=args.tracks_path,
    worker_id=args.worker_id,
    dataset_name=args.dataset_name
)
