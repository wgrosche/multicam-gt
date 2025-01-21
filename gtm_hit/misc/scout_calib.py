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
from typing import Union, List, Optional
from pathlib import Path
from collections import namedtuple

Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])

@dataclass
class Extrinsics:
    R:np.ndarray = None
    T:np.ndarray = None
    def get_R_vec(self):
        return cv.Rodrigues(self.R)[0] if self.R is not None else None
    

@dataclass
class Intrinsics:
    distCoeffs:np.ndarray = None
    cameraMatrix:np.ndarray = None
    newCameraMatrix:np.ndarray = None
    def get_R_vec(self):
        return cv.Rodrigues(self.R)[0] if self.R is not None else None
        
@dataclass
class CameraParams:
    K:np.ndarray = None
    R:np.ndarray = None
    T:np.ndarray = None
    dist:np.ndarray = None
    view_id:Union[int, str] = None
    extrinsics:Extrinsics = None
    intrinsics:Intrinsics = None
    

    

    # def __repr__(self):
    #     return f"Calibration(view_id={str(self.view_id)})\nK:\n{str(self.K)}\nR:\n{str(self.R)}\nT:\n{str(self.T)}\ndist:\n{str(self.dist)}"

    def get_R_vec(self):
        return cv.Rodrigues(self.R)[0] if self.R is not None else None

    def read_from_json(self, filename:Path):
        with open(filename) as f:
            calib_dict = json.load(f)

        self.K =  np.array(calib_dict.get("K", None))
        self.newCameraMatrix =  np.array(calib_dict.get("newCameraMatrix", None))
        self.R =  np.array(calib_dict.get("R", None))
        self.T =  np.array(calib_dict.get("T", None))
        self.dist =  np.array(calib_dict.get("dist", None))
        self.view_id = calib_dict.get("view_id", str(filename.with_suffix("").name))
        self.extrinsics = Extrinsics(self.R, self.T)
        self.intrinsics = Intrinsics(self.dist, self.K, self.newCameraMatrix)

    def as_calib(self):
        return Calibration(self.K, self.R, self.T, self.dist, self.view_id)

        
    def getMaps(self):
        return cv.initUndistortRectifyMap(
            self.K, self.dist, None, self.K, 
            self.size, cv.CV_32FC1
        )
    def set_view_id(self, view_id):
        self.view_id = view_id


def prepare_calibs(calib_source_path, calib_path):
    # symlink calibs to calib_path

    calib_source_path = Path(calib_source_path)
    calib_path = Path(calib_path)

    # create calib_path if it doesn't exist
    calib_path.mkdir(parents=True, exist_ok=True)
    # symlink all calibs in calib_source_path to calib_path
    for calib_file in calib_source_path.glob("*.json"):
        if not (calib_path / calib_file.name).exists():
            (calib_path / calib_file.name).symlink_to(calib_file)

def load_scout_calib(params_dir:Path, cameras:List[str], calib_source_path:Optional[Path] = None):
    # cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    # cam_id_keys = [f"cam_{cam_id[0]}_{cam_id[1]}" for cam_id in cam_id_mat]

    # cam_id_keys_to_idx = dict([cam_id_keys[i],i] 
    #                           for i in range(len(cam_id_keys)))
    # videos_captures = {}
    # output_data = {}
    if calib_source_path is not None and calib_source_path.exists():
        prepare_calibs(calib_source_path, params_dir)
        if not params_dir.exists():
            params_dir.mkdir(parents=True, exist_ok=True)

    cam_params = {}

    for camera_name in cameras:
        camera_parameters = CameraParams(camera_name)
        camera_parameters.read_from_json(params_dir / f"{camera_name}_0.json")
        camera_parameters.set_view_id(camera_parameters)
        cam_params[camera_name] = camera_parameters
        # print("Loaded camera parameters for camera", camera_name, "with", camera_parameters)
    return cam_params