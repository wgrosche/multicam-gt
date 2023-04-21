import frameinfo_pb2 as frameinfo__pb2
from google.protobuf import timestamp_pb2
from google.protobuf.json_format import MessageToDict
import json
import numpy as np
import pandas as pd
import os.path as osp
import datetime

def load_frame_info_chunks(file_path):
    frame_info_list = []

    with open(file_path, "rb") as file:
        while True:
            frame_info = frameinfo__pb2.FrameInfo()
            data = file.read(4)
            if not data:
                break
            size = int.from_bytes(data, byteorder="little")
            frame_info.ParseFromString(file.read(size))
            frame_info_list.append(frame_info)

    return frame_info_list

def frame_info_to_readable_format(frame_info_list):
    readable_frame_info_list = []
    for frame_info in frame_info_list:
        readable_frame_info_list.append(frame_info.timestamp.ToJsonString())
    return readable_frame_info_list

def read_chunks(path):
    cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    cam_id_keys = [f"cam_{cam_id[0]}_{cam_id[1]}" for cam_id in cam_id_mat]
    cams_chunks = {}
    for cam_id in cam_id_mat:
        cam_id = tuple(cam_id)
        cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"
        file_path = osp.join(path,f"cam_{cam_id[0]}_{cam_id[1]}.mp4-frameInfo.chunks")  # Change this to the path of your FrameInfo chunks file
        frame_info_list = load_frame_info_chunks(file_path)
        readable_frame_info_list = frame_info_to_readable_format(frame_info_list)
        cams_chunks[cam_id_key] = readable_frame_info_list
        print("Loaded frame info chunks for camera",cam_id, "for", len(readable_frame_info_list), "frames.")
    return cams_chunks

def parsedate(date):
    return datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ')

def date_diff_in_ms(date1, date2):
    if isinstance(date1, str):
        date1 = parsedate(date1)
    if isinstance(date2, str):
        date2 = parsedate(date2)
    return (date1 - date2).total_seconds() * 1000.0

def get_frames_and_include_frame_drops(chunks_path):
    cams_chunks = read_chunks(chunks_path)
    videos_frames = [v for k,v in cams_chunks.items()]
    difflt = []
    for idx,timestamps in enumerate(pd.DataFrame(videos_frames).T.values.tolist()):
        diffl = []
        for timestamp1 in timestamps:
            for timestamp2 in timestamps:
                diffl.append(int(date_diff_in_ms(timestamp1,timestamp2)))
        diffl = np.array(diffl).reshape(len(timestamps),len(timestamps))
        difflt.append(diffl)
        print(idx)
        print(diffl)
    difflt= np.array(difflt)
    diffdifflt = np.diff(difflt,axis=0)
    #threshold*(no_cams-1)*1/fps
    mask = diffdifflt.sum(axis=2).T>233
    mask_inv = np.logical_not(mask)
    framcnt = mask_inv.cumsum(axis=1)
    framcnt = np.vstack((np.zeros(framcnt.shape[0]),framcnt.T)).astype(int)
    return framcnt

