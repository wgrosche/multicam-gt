import numpy as np
import cv2 as cv
from collections import namedtuple
from enum import IntEnum
from django.conf import settings
from ipdb import set_trace
# import point_cloud_utils as pcu
import trimesh
from .scout_calib import CameraParams

Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'view_id'])


class CuboidVertexEnum(IntEnum):
    FrontTopRight = 0
    FrontTopLeft = 1
    RearTopRight = 2
    RearTopLeft = 3
    FrontBottomRight = 4
    FrontBottomLeft = 5
    RearBottomRight = 6
    RearBottomLeft = 7
    Base = 8
    Direction=9
    CUBOID_VERTEX_COUNT = 10


def get_ray_directions(points_2d, calib):
    points_2d = np.array(points_2d, dtype=float).reshape(-1, 1, 2)

    # Vectorized undistortion
    undistorted_points = cv.undistortPoints(points_2d, calib.K, calib.dist, P=calib.K)
    undistorted_points = undistorted_points.reshape(-1, 2)

    # Homogeneous coordinates
    homogenous = np.hstack([undistorted_points, np.ones((len(undistorted_points), 1))])

    # Precompute matrices
    R_T = calib.R.T
    K_inv = np.linalg.inv(calib.K)
    ray_origin = (-R_T @ calib.T).flatten()

    # Compute ray directions
    temp = K_inv @ homogenous.T
    ray_directions = (R_T @ temp).T

    # Expand ray_origin to match number of points
    ray_origins = np.tile(ray_origin, (len(points_2d), 1))

    return ray_origins, ray_directions


def project_2d_points_to_mesh(points_2d, calib, mesh, VERBOSE=False):
    # Get ray origins and directions
    ray_origins, ray_directions = get_ray_directions(points_2d, calib)
    
    # Perform ray-mesh intersections
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True
    )

    num_rays = len(ray_origins)
    ground_points = np.full((num_rays, 3), None)  # Initialize with None to allow filtering

    # Check if there are no intersections
    if len(locations) == 0:
        if VERBOSE:
            print("No intersections found for any rays.")
        return ground_points#.tolist()

    # Cache variables to reduce attribute lookups
    R, T = calib.R, calib.T.reshape(3, 1)
    camera_coords = - (R @ locations.T) - T
    depths = np.abs(camera_coords[2, :])

    if VERBOSE:
        print(f"Depths sample: {depths[:5]}")

    # Use NumPy to efficiently process the intersections
    min_dist = 1.0
    z_coord_threshold = 0.8

    # Filter intersections by depth and z-coordinates in a single pass
    valid_mask = (depths > min_dist) & (locations[:, 2] < z_coord_threshold)
    
    # Get valid indices per ray
    valid_indices = index_ray[valid_mask]
    valid_depths = depths[valid_mask]
    valid_points = locations[valid_mask]

    # Group and find the closest point for each ray
    if len(valid_indices) > 0:
        unique_rays, inverse_indices = np.unique(valid_indices, return_inverse=True)
        
        # For each unique ray, find the minimum depth and corresponding point
        min_depths_per_ray = np.full(num_rays, np.inf)
        closest_points = np.full((num_rays, 3), None)
        
        for i, ray_idx in enumerate(unique_rays):
            # Get depths and points for the current ray
            ray_depths = valid_depths[inverse_indices == i]
            ray_points = valid_points[inverse_indices == i]

            # Find the closest point based on minimum depth
            closest_idx = np.argmin(ray_depths)
            closest_points[ray_idx] = ray_points[closest_idx]
        
        # Assign only those rays that had valid intersections
        ground_points[unique_rays] = closest_points[unique_rays]
    
    return ground_points#.tolist()


def move_with_mesh_intersection(ground_pix): #reproject to mesh
    """
    Finds the closest point on the mesh to ground_pix
    
    Requires that settings.MESH is a Trimesh.base.Trimesh object
    """

    mesh = settings.MESH
    
    # Use the nearest point function of trimesh
    closest_point, distance, _ = mesh.nearest.on_surface([ground_pix])
    
    # Return the closest point and distance
    return closest_point[0]#, distance[0]


def project_world_to_camera(world_point, K1, R1, T1):
    """
    Project 3D point world coordinate to image plane (pixel coordinate)
    """
    point1 = ((R1 @ world_point) + T1)
    if (np.min(point1[2]) < 0):
        # print("Projection of world point located behind the camera plane")
        return None, None
    point1 = K1 @ point1
    point1 = point1 / point1[2]

    return point1[:2]


def get_cuboid_from_ground_world(world_point:np.ndarray, 
                                 calib:CameraParams, 
                                 height:float, 
                                 width:float, 
                                 length:float, 
                                 theta:float):
    
    cuboid_points3d = np.zeros((CuboidVertexEnum.CUBOID_VERTEX_COUNT, 3))
    cuboid_points3d[CuboidVertexEnum.FrontTopRight] = [width / 2, length / 2, height]
    cuboid_points3d[CuboidVertexEnum.FrontTopLeft] = [-width / 2, length / 2, height]
    cuboid_points3d[CuboidVertexEnum.RearTopRight] = [width / 2, -length / 2, height]
    cuboid_points3d[CuboidVertexEnum.RearTopLeft] = [-width / 2, -length / 2, height]
    cuboid_points3d[CuboidVertexEnum.FrontBottomRight] = [width / 2, length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.FrontBottomLeft] = [-width / 2, length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.RearBottomRight] = [width / 2, -length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.RearBottomLeft] = [-width / 2, -length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.Base] = [0, 0, 0]
    cuboid_points3d[CuboidVertexEnum.Direction] = [0, length / 2, 0]


    rotz = np.array([[np.cos(theta),-np.sin(theta),0],
                     [np.sin(theta), np.cos(theta),0],
                     [            0,             0,1]])
    
    cuboid_points3d = (rotz @ cuboid_points3d.T).T
    cuboid_points3d = cuboid_points3d + world_point.T
    #
    cuboid_points2d = get_projected_points(cuboid_points3d, calib)
    return cuboid_points2d


def get_cuboid2d_from_annotation(annotation, calib, undistort=False):
    #set_trace()
    height= annotation.object_size_x
    width = annotation.object_size_y
    length = annotation.object_size_z
    theta = annotation.rotation_theta
    world_point = annotation.world_point

    # return None if world point is not in fov or too far from camera


    cuboid_points2d = get_cuboid_from_ground_world(world_point, calib, height, width, length, theta)

    return cuboid_points2d


# def get_projected_points(points3d, calib:CameraParams, undistort=False):
#     """
#     Projects points into the image plane.

#     Filters out points that intersect with the mesh on the way to the camera
    
#     """
#     undistort = settings.UNDISTORTED_FRAMES
#     points3d = np.array(points3d).reshape(-1, 3)

#     mesh = settings.MESH



#     if undistort:
#         points2d = project_world_to_camera(points3d, calib.newCameraMatrix, calib.R, calib.T)
#     else:
#         points2d = project_world_to_camera(points3d, calib.K, calib.R, calib.T)

#     points2d = np.squeeze(points2d)
#     points2d = [tuple(p) for p in points2d]
#     return points2d

def get_projected_points(points3d, 
                         calib:CameraParams, 
                         undistort=False):
    """
    Projects points into the image plane and filters non-visible points.
    """
    undistort = settings.UNDISTORTED_FRAMES
    points3d = np.array(points3d).reshape(-1, 3)
    mesh = settings.MESH

    # Get camera position and ray directions
    camera_position = (-calib.R.T @ calib.T).flatten()
    ray_directions = get_ray_directions(points3d, calib)
    
    # Check visibility
    visible_mask = np.ones(len(points3d), dtype=bool)
    
    # Filter points behind camera
    for i, point in enumerate(points3d):
        ray_to_point = point - camera_position
        if np.dot(ray_to_point, calib.R[2]) < 0:  # Check if point is behind camera
            visible_mask[i] = False
            continue
            
        # Check for mesh intersection
        if mesh is not None:
            ray_direction = ray_to_point / np.linalg.norm(ray_to_point)
            ray_origin = camera_position + 0.01 * ray_direction  
            intersections = mesh.ray.intersects_location(ray_origin[None], ray_direction[None])
            if len(intersections[0]) > 0:
                # Compare distance to intersection vs distance to target point
                intersection_dist = np.linalg.norm(intersections[0][0] - ray_origin)
                point_dist = np.linalg.norm(ray_to_point)
                if intersection_dist < point_dist:
                    visible_mask[i] = False

    # Project visible points
    if undistort:
        points2d = project_world_to_camera(points3d[visible_mask], calib.newCameraMatrix, calib.R, calib.T)
    else:
        points2d = project_world_to_camera(points3d[visible_mask], calib.K, calib.R, calib.T)
    if points2d == (None, None):
        raise ValueError("Could not project points to image plane.")
    points2d = np.squeeze(points2d)
    points2d = [tuple(p) for p in points2d]
    return points2d


def get_bounding_box(points):
    points = points.copy()
    try:
        points = np.array(points, dtype=np.int64).reshape(-1, 1, 2)
        x, y, w, h = cv.boundingRect(points)
    except OverflowError:
        return points[0], points[4]
    return  (x, y), (x+w, y+h)
