# from gtm_hit.misc.invision_preprocess import preprocess_invision_data

import sys
from pathlib import Path

# Add the parent directory of this file to sys.path
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.append(str(parent_dir))

from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from argparse import ArgumentParser

parser = ArgumentParser(description="Save multiview data to the annotation tool DB.")

parser.add_argument('-i', "--input_path", default="gtm_hit/labels/json_output", help="Input path to the invision JSON data.")
parser.add_argument('-w',"--worker_id",default=None, help="Worker ID (default: None).")
parser.add_argument('-d',"--dataset_name", default=None, help="Dataset name. It should match the name of the folder in dset/ (default: None).")

parser.add_argument('-t',"--tracks",default="", help="Path to existing tracks, empty by default")
parser.add_argument('-t',"--tracks",default="", help="Path to existing tracks, empty by default")
args = parser.parse_args('-td')





#Note: undistortion and increment is determined by the settings file.
class Args:
    def __init__(self,
                #  frames_path="",
                #  calibration_path="",
                 tracks_path="",
                #  input_path="gtm_hit/static/gtm_hit/labels/json_output",
                 worker_id=None, #settings.WORKER_ID,
                 hdf5_template = "/cvlabdata2/home/grosche/dev/calibration/sync_frame_seq_1/{camera}",
                #  hdf5_template = "/cvlabscratch/home/engilber/dev/calibration/data/calib_test_2/initial_calibration/{camera}",
                 dataset_name=settings.DSETNAME, 
                 range_start=settings.FRAME_START,
                 range_end=settings.FRAME_END,
                 dict_path =''):
        
        # self.frames_path=frames_path,
        # self.calibration_path=calibration_path,
        self.tracks_path=tracks_path,
        # self.input_path = input_path
        self.worker_id = worker_id
        self.dataset_name = dataset_name
        self.range_start=range_start
        self.range_end=range_end
        self.hdf5_template = hdf5_template
        self.dict_path = dict_path



if __name__ == "__main__":

    args = Args()
    args.dict_path = "/cvlabdata2/home/grosche/dev/calib/trajectories_weighted_mean_ground.json"
    args.worker_id = "SCIPIO"
    preprocess_scout_data_from_dict(
        hdf5_template = "/cvlabdata2/home/grosche/dev/calibration/sync_frame_seq_1/{camera}", 
        worker_id=args.get("worker_id", settings.WORKER_ID),
        dataset_name=args.get("dataset_name", settings.DSETNAME), 
        dict_path = args.dict_path
        )