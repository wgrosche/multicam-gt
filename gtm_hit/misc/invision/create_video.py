#video writer
import os
from django.db import models, transaction
import numpy as np
from django.conf import settings
from gtm_hit.misc import geometry
from gtm_hit.models import Annotation, Annotation2DView, MultiViewFrame, Person, View
from django.core.exceptions import ObjectDoesNotExist
from ipdb import set_trace
from gtm_hit.misc.invision.camera_frame import CameraFrame
import cv2 as cv

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-2], images.shape[-3]

    if grid_size is not None:
        grid_h, grid_w = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros([grid_h * img_h, grid_w * img_w] + list(images.shape[-1:]), dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y : y + img_h, x : x + img_w, ...] = images[idx]
    return grid

def create_video(video_out, fps, dataset_name, worker_id):
    set_trace()
    video_writer = None
    break_flag = False
    cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    for i, cam_id in enumerate(cam_id_mat):
        cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"
        with open(f'motmetrics/gt_{cam_id_key}_output.txt', 'w') as f:
        
            for frame_id in range(3150, 4415,7):
                frame_list = []
                try:
                    frame = MultiViewFrame.objects.get(frame_id=frame_id, dataset__name=dataset_name,worker_id=worker_id)
                except ObjectDoesNotExist:
                    continue
                print(f'[{cam_id_key}] Processing frame {frame_id} ...', end='\r')

                annotations2d_cam = Annotation2DView.objects.filter(annotation__frame=frame, view__view_id=i)#annotation__person__annotation_complete=True)
                
                for det in annotations2d_cam:
                    if det.cuboid_points==None:
                        continue

                    track_id = det.annotation.person.person_id

                    bb_left = det.x1
                    bb_top = det.y1
                    bb_width = det.x2 - det.x1
                    bb_height = det.y2 - det.y1

                    confidence = 1

                    x,y,z = det.annotation.Xw, det.annotation.Yw, det.annotation.Zw

                    
                    annotation_complete = det.annotation.person.annotation_complete
                    if annotation_complete:
                        line = f"{frame_id},{track_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{confidence},{x:.2f},{y:.2f},{z:.2f}\n"

                        f.write(line)
            print(f'[{cam_id_key}] OK ...')
        print(f'Done!')
        

def create_video2(video_out, fps, dataset_name, worker_id):
    video_writer = None
    break_flag = False
    cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    for frame_id in range(3150, 5000,7):
        frame_list = []
        try:
            frame = MultiViewFrame.objects.get(frame_id=frame_id, dataset__name=dataset_name,worker_id=worker_id)
        except ObjectDoesNotExist:
            continue
        print(f'Processing frame {frame_id}...')

        for i, cam_id in enumerate(cam_id_mat):
            annotations2d_cam = Annotation2DView.objects.filter(annotation__frame=frame, view__view_id=i)#annotation__person__annotation_complete=True)
            cam_id = tuple(cam_id)

            framepath = f"gtm_hit/static/gtm_hit/dset/13apr/undistorted_frames/cam{i+1}/{frame_id:08d}.jpg"
            camera_frame = CameraFrame(framepath, None, None, is_distorted=False)
            framesave = camera_frame.get_frame_with_db_annotations(annotations2d_cam)
            if not os.path.exists(f"13apr/undistorted_frames/cam{i+1}/"):
                os.makedirs(f"13apr/undistorted_frames/cam{i+1}/")
            # ret = cv.imwrite(
            #     f"13apr/undistorted_frames/cam{i+1}/{frame_id:08d}.jpg", framesave)
            frame_list.append(framesave)
            
        
        frame_list = np.array(frame_list)
        frames_grid = create_image_grid(frame_list, grid_size=(3, 3))
        out_size = frames_grid.shape[0:2]
        out_size = out_size[::-1]
        if video_writer is None:
            video_writer = cv.VideoWriter(video_out,
                                    cv.VideoWriter_fourcc(*'XVID'),
                                    fps,
                                    out_size)
        
        video_writer.write(frames_grid)

        if break_flag:
            break
    video_writer.release()