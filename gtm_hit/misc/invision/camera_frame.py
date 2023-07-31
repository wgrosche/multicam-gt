
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import glob
import json
import sys
import pandas as pd

class CameraFrame:
    def __init__(self, frame = None, output: dict = None , camera_params= None, is_distorted: bool = True, apply_undistortion: bool = True, median_frame = None):
        if isinstance(frame, np.ndarray):
            self.frame = frame
        elif isinstance(frame, str):
            self.frame = cv.imread(frame,cv.COLOR_RGB2BGR)
        self.frameId = output.get('frameId',None)
        self.timestamp = output.get('timestamp',None)
        self.network_detections = output.get('network_detections',None)
        self.tracked_detections = output.get('tracked_locations',None)
        self.camera_params = camera_params
        self.is_distorted = is_distorted
        if median_frame is not None:
            self.frame = self.frame - median_frame #subtract median frame and convert all negative values to zero
            self.frame[self.frame<0] = 0
            
        if is_distorted and apply_undistortion:
            self.undistort()
            self.is_distorted = False
            
        self.load_detections()
        
            
    
    def __repr__(self) -> str:
        return f"CameraFrame\
                (timestamp={self.timestamp},\
                frameId={self.frameId}, \
                is_distorted={self.is_distorted}, \
                no_network_detections={len(self.network_detections) if self.network_detections is not None else 0}, \
                no_tracked_detections={len(self.tracked_detections) if self.tracked_detections is not None else 0})"
    def undistort(self):
        mapx, mapy = self.camera_params.getMaps()
        self.frame = cv.remap(self.frame, mapx, mapy, cv.INTER_LINEAR)
    def load_detections(self):
        if self.network_detections is not None:
            self.network_detections = [NetworkDetection(det) for det in self.network_detections]
        if self.tracked_detections is not None:
            self.tracked_detections = [TrackedDetection(det) for det in self.tracked_detections]
            self.tracked_detections_dict = dict([(det.track_id, det) for det in self.tracked_detections])

    def get_frame(self, display_network_detections = False, display_tracked_detections = True):
        frame_copy = self.frame.copy()
        if self.is_distorted:
            if display_network_detections and self.network_detections is not None:
                for ndet in self.network_detections:
                    pass
                    #todo finish
            if display_tracked_detections and self.tracked_detections is not None:
                for tdet in self.tracked_detections:
                    
                    cuboid_points3d = tdet.get_cuboid_world_vertices()
                    cuboid_points2d = self.get_projected_points(cuboid_points3d)
                    
                    plot_cuboid(frame_copy, cuboid_points2d)

                    p1,p2 = get_bounding_box(cuboid_points2d)
                    plot_bounding_box(frame_copy, p1,p2,RED,name=f"ID:{tdet.track_id}") #plot cuboid bb
                    


        else: #undistorted
            if display_tracked_detections and self.tracked_detections is not None:
                for tdet in self.tracked_detections:
                    
                    cuboid_points2d = np.array(tdet.cuboid).reshape(-1,2)
                    cuboid_points2d = [tuple(p) for p in cuboid_points2d]
                    cuboid_points2d = cuboid_points2d[:4]+ [cuboid_points2d[6],cuboid_points2d[4],cuboid_points2d[5],cuboid_points2d[7]] #correct the order for plotting
                    
                    plot_cuboid(frame_copy, cuboid_points2d,PURPLE)
                    
                    # p1,p2 = get_bounding_box(cuboid_points2d) #compute cuboid bbox
                    # plot_bounding_box(frame_copy, p1,p2,PINK,name=f"ID:{tdet.track_id}") #plot cuboid bbox


                    #NEW: get projected with undistortion
                    cuboid_points3d = tdet.get_cuboid_world_vertices()
                    cuboid_points2d = self.get_projected_points(cuboid_points3d,undistort=True)
                    
                    plot_cuboid(frame_copy, cuboid_points2d,GREEN)
                    
                    p1,p2 = get_bounding_box(cuboid_points2d) #compute cuboid bbox
                    plot_bounding_box(frame_copy, p1,p2,PINK,name=f"ID:{tdet.track_id}") #plot cuboid bbox

                    # p1,p2 = tdet.get_bbox_points()
                    # plot_bounding_box(frame_copy, p1,p2,PINK,name=f"ID:{tdet.track_id}") #plot precomputed bbox

        return frame_copy
    def get_tracked_person(self, track_id,return_points=False):
        frame_copy = self.frame.copy()
        if self.tracked_detections is not None:
            if track_id in self.tracked_detections_dict:
                tdet = self.tracked_detections_dict[track_id]
                if self.is_distorted:
                    cuboid_points3d = tdet.get_cuboid_world_vertices()
                    cuboid_points2d = self.get_projected_points(cuboid_points3d)

                    p1,p2 = get_bounding_box(cuboid_points2d)
                            
                else: #undistorted
                    cuboid_points2d = np.array(tdet.cuboid).reshape(-1,2)
                    cuboid_points2d = [tuple(p) for p in cuboid_points2d]

                    p1,p2 = get_bounding_box(cuboid_points2d) #compute cuboid bbox
                    # p1,p2 = tdet.get_bbox_points() #precomputed bbox points
            else:
                return None
        else:
            return None
        
        if return_points:
            return p1,p2
        
        x1,y1 = p1
        x2,y2 = p2

        return frame_copy[y1:y2,x1:x2]
    

    ##delete?
    def get_tracked_all(self,return_points=False):
        frame_copy = self.frame.copy()
        bboxes = []

        if self.tracked_detections is not None:
            for tdet in self.tracked_detections:
                if self.is_distorted:
                    cuboid_points3d = tdet.get_cuboid_world_vertices()
                    cuboid_points2d = self.get_projected_points(cuboid_points3d)

                    p1,p2 = get_bounding_box(cuboid_points2d)
                    
                else: #undistorted
                    cuboid_points2d = np.array(tdet.cuboid).reshape(-1,2)
                    cuboid_points2d = [tuple(p) for p in cuboid_points2d]

                    #p1,p2 = get_bounding_box(cuboid_points2d) #compute cuboid bbox
                    p1,p2 = tdet.get_bbox_points() #precomputed bbox points


                if return_points:
                    bboxes.append((p1,p2))
                else:
                    frame_copy = self.frame.copy()
                    x1,y1 = p1
                    x2,y2 = p2
                    bboxes.append(frame_copy[y1:y2,x1:x2])
        else:
            return None
        
        return bboxes
    
    def show_frame(self, display_network_detections = False, display_tracked_detections = True):
        frame_copy = self.get_frame(display_network_detections, display_tracked_detections)
        if frame_copy is None:
            return None
        imshow(frame_copy,figsize=(20,20))
        
    def show_tracked_person(self, track_id):
        frame_copy = self.get_tracked_person(track_id)
        imshow(frame_copy,figsize=(20,20))
        
    def get_projected_points(self,points3d,undistort=False):
        points3d = np.array(points3d).reshape(-1,3)
        Rvec = self.camera_params.extrinsics.get_R_vec() #cv.Rodrigues
        Tvec = self.camera_params.extrinsics.T
        points2d,_ = cv.projectPoints(points3d,Rvec,Tvec,self.camera_params.cameraMatrix,self.camera_params.distCoeffs)
        if undistort:
            #points3d_homogenous = np.hstack([points3d,np.ones((points3d.shape[0],1))])
            points3d_cam = self.camera_params.extrinsics.R @ points3d.T + self.camera_params.extrinsics.T.reshape(-1,1)
            points3d_cam_rectified = self.camera_params.Rmat @ points3d_cam #correct the slant of the camera
            points2d = self.camera_params.newCameraMatrix @ points3d_cam_rectified

            points2d = points2d[:2,:]/points2d[2,:]
            points2d = points2d.T
        points2d = np.squeeze(points2d)
        points2d = [tuple(p) for p in points2d]
        return points2d

def plot_cuboid(img, vertices, color=(255, 255, 0), scale=1):
    vertices = vertices.copy()
    for idx,point in enumerate(vertices):
        if point:
            point = tuple([int(p*scale) for p in point])
    line_1 = [0, 1, 5, 4, 4, 0, 1, 5, 7, 3, 2, 6]
    line_2 = [1, 5, 4, 0, 7, 3, 2, 6, 3, 2, 6, 7]
    for p1, p2 in zip(line_1, line_2):
        if vertices[p1] and vertices[p2]:
            _p1 = tuple([int(p * scale) for p in vertices[p1]])
            _p2 = tuple([int(p * scale) for p in vertices[p2]])
            img = cv.line(img, _p1, _p2, color=color, thickness=1)

def get_bounding_box(points):
    points = points.copy()
    points = np.array(points,dtype=np.int32).reshape(-1,1,2)
    x,y,w,h = cv.boundingRect(points)
    return (x,y),(x+w,y+h)

def plot_bounding_box(img, p1,p2,color=(0, 0, 255),thickness=2, name=None):
    if name is not None:
        # print(p1,p2)
        # print(name)
        textLoc = (p1[0]+(p2[0]-p1[0])/2, p2[1]-20)
        textLoc = (int(textLoc[0]), int(textLoc[1]))
        draw_text(img, name, textLoc)  
    cv.rectangle(img, p1, p2, color, thickness)

def draw_text(img, text,
          pos=(0, 0),
          font=cv.FONT_HERSHEY_PLAIN,
          font_scale=1.5,
          text_color=(0, 0, 0),
          font_thickness=1,
          text_color_bg=(0, 0, 0)
          ):
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    #cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    if y < 0 or x < 0 or y + text_h > img.shape[0] or x + text_w > img.shape[1]:
        return text_size
    sub_img = img[y:y+text_h, x:x+text_w]

    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    # Putting the image back to its position
    img[y:y+text_h, x:x+text_w] = res
    cv.putText(img, text, (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness, cv.LINE_AA)
    return text_size

class Detection:
    def __init__(self, bbox, class_name, cuboid):
        self.bbox = bbox
        self.class_name = class_name
        self.cuboid = cuboid
    
    def __init__(self, detection_dict:dict):
        self.bbox = detection_dict["bbox"]
        self.class_name = detection_dict["class"]
        self.cuboid=detection_dict["cuboid"]
    
    def get_bbox_points(self):
        x1,y1,x2,y2 = [int(x) for x in self.bbox]
        return (x1,y1),(x2,y2)

class NetworkDetection(Detection):
    def __init__(self,bbox, class_name, cuboid, score):
        super().__init__(bbox, class_name, cuboid)
        self.score = score
    def __init__(self, detection_dict):
        self.bbox = detection_dict["bbox"]
        self.class_name = detection_dict["class"]
        self.cuboid=detection_dict["cuboid"]
        self.score = detection_dict["ml_score"]

class TrackedDetection(Detection):
    def __init__(self,bbox, class_name, cuboid, score, cuboid_to_world_transform,detection_associated_idx,object_size, track_id, uncertainty_ellipse_m2):
        super().__init__(bbox, class_name, cuboid, score)
        self.cuboid_to_world_transform = cuboid_to_world_transform
        self.detected_associated_idx = detection_associated_idx
        self.object_size = object_size
        self.track_id = track_id
        self.uncertainty_ellipse_m2 = uncertainty_ellipse_m2
        self.compute_cuboid()
    def __repr__(self) -> str:
        return f"TrackedDetection({self.__dict__})"
    def __init__(self, tracked_location_dict):
        super().__init__(tracked_location_dict)
        #self.cuboid_to_world_transform = tracked_location_dict["cuboidToWorldTransform"]
        self.cuboid_to_world_transform =np.array(tracked_location_dict["object_to_world"],dtype=np.float32) 

        self.detected_associated_idx = tracked_location_dict["detection_associated_idx"]
        # self.object_size = tracked_location_dict["objectSize"]
        self.object_size = [2,0.5,0.5]
        self.track_id = tracked_location_dict["trackId"]
        self.uncertainty_ellipse_m2 = tracked_location_dict["uncertainty_ellipse_m2"]
        self.compute_cuboid()

    def compute_cuboid(self):
        self.cuboid_obj = Cuboid(self.object_size)
        self._cuboid_world_vertices = self.cuboid_obj.get_world_vertices(self.cuboid_to_world_transform)
    def get_cuboid_world_vertices(self):
        return self._cuboid_world_vertices
    
    def get_base_in_world(self):
        return self.cuboid_obj.get_base_in_world(self.cuboid_to_world_transform)

class Cuboid:
    def __init__(self, object_size):
        self.object_size = object_size
        self.vertices = np.zeros((CUBOID_VERTEX_COUNT, 3))
        self.gen_vertices()
    def __repr__(self) -> str:
        return "Cuboid({})".format(self.vertices)
    def gen_vertices(self):
        width, length, height = self.object_size[::-1]
        self.vertices[CuboidVertexEnum.FrontTopRight] = [width / 2, length / 2, height]
        self.vertices[CuboidVertexEnum.FrontTopLeft] = [-width / 2, length / 2, height]
        self.vertices[CuboidVertexEnum.FrontBottomLeft] = [-width / 2, -length / 2, height]
        self.vertices[CuboidVertexEnum.FrontBottomRight] = [width / 2, -length / 2, height]
        self.vertices[CuboidVertexEnum.RearTopRight] = [width / 2, length / 2, 0]
        self.vertices[CuboidVertexEnum.RearTopLeft] = [-width / 2, length / 2, 0]
        self.vertices[CuboidVertexEnum.RearBottomLeft] = [-width / 2, -length / 2, 0]
        self.vertices[CuboidVertexEnum.RearBottomRight] = [width / 2, -length / 2, 0]
        self.base = np.zeros((1, 3))
    
    def get_world_vertices(self, cuboid_to_world_transform):
        vertices = np.hstack([self.vertices,np.ones((8,1))])
        return (cuboid_to_world_transform @ vertices.T).T[:,:3]

    def get_base_in_world(self, cuboid_to_world_transform):
        base = np.hstack([self.base,np.ones((1,1))])
        return (cuboid_to_world_transform @ base.T).T[:,:3]

PINK = (250,190,203)
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
YELLOW = (0,255,255)
ORANGE = (0,140,255)
PURPLE = (255,0,255)

from enum import IntEnum

class CuboidVertexEnum(IntEnum):
    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    Base = 8
CUBOID_VERTEX_COUNT = 8
        