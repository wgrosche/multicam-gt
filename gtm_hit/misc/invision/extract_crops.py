#video writer
import os
import numpy as np
from gtm_hit.models import Annotation2DView, MultiViewFrame
from django.core.exceptions import ObjectDoesNotExist
from ipdb import set_trace
from gtm_hit.misc.invision.camera_frame import CameraFrame
import cv2 as cv
import os.path as osp
import os
from shutil import copyfile
from random import randint
from IPython.display import clear_output

def extract_crops(dataset_name, worker_id,nested=True):
    set_trace()
    root = osp.join("gtm_hit", "static", "gtm_hit", "dset")
    dataset_path = osp.join(root, dataset_name)

    break_flag = False
    cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    for frame_id in range(3150, 5000,7):
        
        try:
            frame = MultiViewFrame.objects.get(frame_id=frame_id, dataset__name=dataset_name,worker_id=worker_id)
        except ObjectDoesNotExist:
            continue
        print(f'Processing frame {frame_id}...')

        for i, cam_id in enumerate(cam_id_mat):
            annotations2d_cam = Annotation2DView.objects.filter(annotation__frame=frame, view__view_id=i)#annotation__person__annotation_complete=True)
            cam_id = tuple(cam_id)
            
            for annotation in annotations2d_cam:
                annotation_person = annotation.annotation.person
                person_id = annotation_person.person_id
                if annotation_person.annotation_complete:

                    framepath = osp.join(dataset_path, "undistorted_frames", f"cam{i+1}/{frame_id:08d}.jpg")
                    camera_frame = CameraFrame(framepath, None, None, is_distorted=False)
                    crop = camera_frame.get_crop_from_db_annotation(annotation)

                    if crop is not None and crop.any():
                        crop_save_path = osp.join(dataset_path, "crops",worker_id)
                        
                        if nested:
                            if not osp.exists(osp.join(crop_save_path,f"cam_{cam_id[0]}_{cam_id[1]}",f"{person_id}")):
                                os.makedirs(osp.join(crop_save_path,f"cam_{cam_id[0]}_{cam_id[1]}",f"{person_id}"), exist_ok=True)
            
                            ret = cv.imwrite(osp.join(crop_save_path,f"cam_{cam_id[0]}_{cam_id[1]}",f"{person_id}",f"frame_{frame_id}_person_{person_id}.jpg"),crop)
                        else:
                            if not osp.exists(crop_save_path):
                                os.makedirs(crop_save_path, exist_ok=True)

                            ret = cv.imwrite(osp.join(crop_save_path, f"id{person_id}_cam{cam_id[0]}{cam_id[1]}_frame{frame_id}.jpg"),crop)

                        if not ret:
                            print(f"{worker_id}@{dataset_name} [FRAME:{frame_id:5d} CAM{(i+1):2d} pID:{person_id:5d}] ERROR: Could not write crop to disk.")
                        else:
                            print(f"{worker_id}@{dataset_name} [FRAME:{frame_id:5d} CAM{(i+1):2d} pID:{person_id:5d}] OK")
        if break_flag:
            break


def prepare_pytorch(dataset_name, worker_id):
    #requires crops to be extracted and nested

    root = osp.join("gtm_hit", "static", "gtm_hit", "dset")
    dataset_path = osp.join(root, dataset_name)
    download_path = osp.join(dataset_path, "crops",worker_id)
    dst_root_path = osp.join(dataset_path, "pytorch",worker_id,"pytorch")
    set_trace()
    query_save_path = os.path.join(dst_root_path,"query")
    if not os.path.isdir(query_save_path):
        os.makedirs(query_save_path)
    
    for root, dirs, files in os.walk(download_path, topdown=True):
        #['.', 'footage', 'frames', 'cam_1_1', '986']
        if len(files)==0:
            continue
        cam_id, person_id = root.split("/")[-2:]
        cam_id = cam_id.replace("_","")
        print(cam_id,person_id)
        clear_output(wait=True)
        #get query
        query_idx = randint(0,len(files)-1)
        query_name = files[query_idx]
        query_frame = int(query_name.split('_')[1])
        files.pop(query_idx)
        src_path = os.path.join(root,query_name)

        query_person_save_path = os.path.join(query_save_path,person_id)
        if not os.path.isdir(query_person_save_path):
            os.makedirs(query_person_save_path)
        copyfile(src_path, os.path.join(query_person_save_path,f"{person_id}_{cam_id}_{query_frame}.jpg"))
        
        
        # train_test_files = []
        # last_frame = None
        # for name in sorted(files):
        #     if not name[-3:]=='jpg':
        #         continue
        #     current_frame = int(name.split('_')[1])
        #     if last_frame is None:
        #         last_frame = current_frame
        #         train_test_files.append(name)
        #     # if (current_frame - last_frame) < 5: #skip 5 frames
        #     #     continue
        #     train_test_files.append(name)
        #     last_frame = current_frame
        # #test train split
        # if len(train_test_files) < 10:
        #     continue

        #split_idx = int(len(train_test_files)*0.3)
        #train_files = train_test_files[split_idx:]
        #test_files = train_test_files[:split_idx]

        test_files = files

        for mode,files in zip(["gallery"],[test_files]): #zip(["train_all","gallery"],[train_files,test_files]):
            for name in files:
                src_path = osp.join(root,name)
                dst_path = osp.join(dst_root_path,mode,person_id)

                if not os.path.isdir(dst_path):
                    os.makedirs(dst_path)

                current_frame = int(name.split('_')[1])
                copyfile(src_path, f"{dst_path}/{person_id}_{cam_id}_{current_frame}.jpg")

                if mode == "train_all": #add train/val split additionally
                    dst_path = dst_root_path + f"train/{person_id}"
                    if not os.path.isdir(dst_path):
                        os.makedirs(dst_path)
                        dst_path = dst_root_path + f"val/{person_id}"  #first image is used as val image
                        os.makedirs(dst_path)
                    copyfile(src_path, f"{dst_path}/{person_id}_{cam_id}_{current_frame}.jpg")
