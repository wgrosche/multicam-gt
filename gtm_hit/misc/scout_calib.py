import cv2 as cv
import numpy as np
import os
import glob
import json
import sys
import os.path as osp
import numpy as np
from ipdb import set_trace
from dataclasses import dataclass
from typing import Union, List
from pathlib import Path
from collections import namedtuple

Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])

@dataclass
class CameraParams:
    K:np.ndarray = None
    R:np.ndarray = None
    T:np.ndarray = None
    dist:np.ndarray = None
    view_id:Union[int, str] = None

    def __repr__(self):
        return f"Calibration(view_id={self.view_id})\nK:\n{self.K}\nR:\n{self.R}\nT:\n{self.T}\ndist:\n{self.dist}"

    def get_R_vec(self):
        return cv.Rodrigues(self.R)[0] if self.R is not None else None

    def read_from_json(self, filename:Path):
        with open(filename) as f:
            calib_dict = json.load(f)

        self.K =  np.array(calib_dict.get("K", None))
        self.R =  np.array(calib_dict.get("R", None))
        self.T =  np.array(calib_dict.get("T", None))
        self.dist =  np.array(calib_dict.get("dist", None))
        self.view_id = calib_dict.get("view_id", str(filename.with_suffix("").name))

    def as_calib(self):
        return Calibration(self.K, self.R, self.T, self.dist, self.view_id)

        
    def getMaps(self):
        return cv.initUndistortRectifyMap(
            self.K, self.dist, None, self.K, 
            self.size, cv.CV_32FC1
        )
    def set_view_id(self, view_id):
        self.view_id = view_id

def load_scout_calib(params_dir:Path, cameras:List[str]):
    # cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    # cam_id_keys = [f"cam_{cam_id[0]}_{cam_id[1]}" for cam_id in cam_id_mat]

    # cam_id_keys_to_idx = dict([cam_id_keys[i],i] 
    #                           for i in range(len(cam_id_keys)))
    # videos_captures = {}
    # output_data = {}
    cam_params = {}

    for camera_name in cameras:
        camera_parameters = CameraParams(camera_name)
        camera_parameters.read_from_json(params_dir / f"{camera_name}.json")
        camera_parameters.set_view_id(camera_parameters)
        cam_params[camera_name] = camera_parameters
        print("Loaded camera parameters for camera ", camera_name,".")
    return cam_params