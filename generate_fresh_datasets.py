from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
# import argparse
# parser = argparse.ArgumentParser(description="Save invision data to the annotation tool DB.")
# parser.add_argument('-i', "--input_path", default="gtm_hit/labels/json_output", help="Input path to the invision JSON data.")
# parser.add_argument('-w',"--worker_id",default="INVISION1", help="Worker ID (default: INVISION1).")
# parser.add_argument('-d',"--dataset_name", default="invision", help="Dataset name. It should match the name of the folder in dset/ (default: invision).")
# args = parser.parse_args()


merging_strategies_to_generate = ['mean']#['unmerged', 'mean', 'camera_mean_top', 'camera_mean']
worker_names = ['WILKEMEAN']#['UNMERGED2', 'MEAN2', 'CAMERAMEANTOP2', 'CAMERAMEAN2']

#Note: undistortion and increment is determined by the settings file.
class Args:
    def __init__(self,
                #  frames_path="",
                #  calibration_path="",
                 tracks_path="",
                #  input_path="gtm_hit/static/gtm_hit/labels/json_output",
                 worker_id=settings.WORKER_ID,
                 hdf5_template = "/cvlabdata2/home/grosche/dev/calibration/sync_frame_seq_1/{camera}",
                 dataset_name=settings.DSETNAME, 
                 range_start=settings.FRAME_START,
                 range_end=settings.FRAME_END,
                 dict_path ='/cvlabdata2/home/grosche/dev/calibration/traj_dict.json'):
        
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

args = Args()
# args.frames_path=Path("/cvlabscratch/home/engilber/datasets/SCOUT/collect_30_05_2024/sync_frame_seq_1/")
# args.calibration_path=Path("/cvlabdata2/home/grosche/dev/calibration/calibrations")
args.tracks_path=Path("/cvlabdata2/home/grosche/dev/calibration/unmerged_tracks.pkl")

# preprocess_scout_data(
#     tracks_path=args.tracks_path,
#     worker_id=args.worker_id,
#     dataset_name=args.dataset_name,
#     range_start=args.range_start,
#     range_end=args.range_end,
#     testing = False
# )
for strategy in merging_strategies_to_generate:
    print("Generating dataset for strategy: ", strategy)
    dict_path = Path('/cvlabdata2/home/grosche/dev/calibration/') / f'traj_dict_{strategy}.json'
    # preprocess_scout_data_from_dict(args.hdf5_template,
    #                       worker_id=worker_names[merging_strategies_to_generate.index(strategy)], 
    #                       dataset_name=args.dataset_name,
    #                       dict_path=dict_path
    #                       )
    preprocess_scout_data_from_dict(args.hdf5_template,
                          worker_id=worker_names[merging_strategies_to_generate.index(strategy)], 
                          dataset_name=args.dataset_name,
                          dict_path=dict_path
                          )