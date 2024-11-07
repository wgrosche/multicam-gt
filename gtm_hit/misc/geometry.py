import numpy as np
import cv2 as cv
from collections import namedtuple
from enum import IntEnum
from django.conf import settings
from ipdb import set_trace
# import point_cloud_utils as pcu
import trimesh
from .scout_calib import CameraParams
from matplotlib.path import Path as mplpath

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
    if settings.FLAT_GROUND:
        return ground_pix
    else:
        mesh = settings.MESH
        
        # Use the nearest point function of trimesh
        closest_point, distance, _ = mesh.nearest.on_surface(ground_pix.reshape(-1, 3))
        ground_pixel = closest_point[0]
    # Return the closest point and distance
    return ground_pixel


def project_world_to_camera(world_point, K1, R1, T1):
    """
    Project 3D point world coordinate to image plane (pixel coordinate)
    """
    point1 = ((R1 @ world_point.reshape(3, 1)) + T1).reshape(3, 1)
    point1 = K1 @ point1
    point1 = point1 / point1[2]

    return point1[:2]

def get_projected_points(points3d, 
                         calib:CameraParams, 
                         undistort=False):
    """
    Projects points into the image plane and filters non-visible points.
    """
    undistort = settings.UNDISTORTED_FRAMES
    points3d = np.array(points3d).reshape(-1, 3)
    # Project visible points
    if undistort:
        points2d = [project_world_to_camera(point3d, calib.newCameraMatrix, calib.R, calib.T) for point3d in points3d]
    else:
        points2d = [project_world_to_camera(point3d, calib.K, calib.R, calib.T) for point3d in points3d]
    # if points2d == (None, None):
    #     raise ValueError("Could not project points to image plane.")
    points2d = np.squeeze(points2d)
    points2d = [tuple(p) for p in points2d]
    return points2d


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


def get_cuboid2d_from_annotation(annotation, cam_name, undistort=False):
    calib = settings.CALIBS[cam_name]
    #set_trace()
    height= annotation.object_size_x
    width = annotation.object_size_y
    length = annotation.object_size_z
    theta = annotation.rotation_theta
    world_point = annotation.world_point
    # print("Processing cuboid: ", world_point, height, width, length, theta)
    # check that world point is visible
    if world_point is None or np.any(np.isnan(world_point)):
        return None
    
    # check that world point is in fov
    if not is_visible(world_point, cam_name, check_mesh=True):
        return None
    
    # print("adding cuboid at: ", world_point)


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


from shapely.geometry import Point, Polygon


def get_polygon_from_points_3d(points_3d):
    polygon_points = [(point[0], point[1]) for point in points_3d]
    return Polygon(polygon_points)


def is_point_in_polygon(polygon, test_point):
    # print(test_point)
    test_point = np.array(test_point).flatten()
    test_point_2d = Point(test_point[0], test_point[1])
    # Check if point is inside or on boundary
    return polygon.contains(test_point_2d) or polygon.touches(test_point_2d)

def is_visible(point3d:np.ndarray, cam_name:str, check_mesh:bool = True) -> bool:
    """
    Checks if a 3D point is visible in the camera frame.
    """
    calib = settings.CALIBS[cam_name]
    polygon = settings.ROI[cam_name]
    point3d = np.array(point3d).reshape(-1, 3)
    mesh = settings.MESH
    camera_position = (-calib.R.T @ calib.T).flatten()
    ray_to_point = point3d - camera_position
    # ray_origins, ray_directions = get_ray_directions(point3d, calib)
    # ray_direction = ray_to_point / np.linalg.norm(ray_to_point)

    # check behind camera
    if np.dot(ray_to_point, calib.R[2]) < 0:
        return False
    
    # check if point in ROI
    if not is_point_in_polygon(polygon, point3d):
        # print("Point not in ROI")
        return False
    
    print("Point in ROI")
    
    # Check if there’s an intersection between the ray and the mesh
    # if mesh is not None and check_mesh:
    #     # Get intersection locations with the mesh
    #     locations, _, _ = mesh.ray.intersects_location(
    #         ray_origins=np.array(camera_position), 
    #         ray_directions=np.array(ray_direction)
    #     )
        
    #     if len(locations) > 0:
    #         # Use squared distances to save computation
    #         distance_to_point_sq = np.dot(ray_to_point, ray_to_point)
    #         distance_to_intersection_sq = min(
    #             np.dot(location - camera_position, location - camera_position) 
    #             for location in locations
    #         )

    #         # If there’s an intersection closer than the point, return False
    #         if distance_to_intersection_sq < distance_to_point_sq:
    #             print(f"Point {point3d} is not visible due to mesh intersection.")
    #             return False

            
    # If no intersection occurs before the point, return True
    return True




def get_bounding_box(points):
    points = points.copy()
    try:
        points = np.array(points, dtype=np.int64).reshape(-1, 1, 2)
        x, y, w, h = cv.boundingRect(points)
    except OverflowError:
        return points[0], points[4]
    return  (x, y), (x+w, y+h)


def reproject_to_world_ground_batched(ground_pix, K0, R0, T0, height=0):
    """
    Compute world coordinates from pixel coordinates of points on a plane at specified height
    
    Args:
    ground_pix (array-like): The pixel coordinates of points in the image. Shape (N,2) or (N,3)
    K0 (array-like): The camera intrinsic matrix 
    R0 (array-like): The camera rotation matrix
    T0 (array-like): The camera translation vector
    height (float): The height of the plane in world coordinates (default: 0)
    
    Returns:
    array-like: The 3D world coordinates of the points. Shape (N,3)
    """
    # print("ground_pix: ", np.array(ground_pix).shape)
    # print("K0: ", np.array(K0).shape)
    # print("R0: ", np.array(R0).shape)
    # print("T0: ", np.array(T0).shape)
    # Convert ground_pix to homogeneous coordinates if needed
    if ground_pix.shape[1] == 2:
        ground_pix_hom = np.hstack((ground_pix, np.ones((ground_pix.shape[0], 1))))
    else:
        ground_pix_hom = ground_pix
        
    # print("ground_pix_hom: ", np.array(ground_pix_hom).shape)
    # Transpose to 3xN for matrix operations
    ground_pix_hom = ground_pix_hom.T
        
    C0 = -R0.T @ T0
    
    K0_inv = np.linalg.inv(K0)
    
    l = R0.T @ K0_inv @ ground_pix_hom
    
    # Calculate the scaling factor to reach the specified height
    scale = (height - C0[2]) / l[2,:]
    
    # Reshape scale to broadcast correctly
    scale = scale.reshape(-1)#[np.newaxis,:]
    # print("scale: ", np.array(scale).shape)
    
    # Broadcast C0 to match dimensions
    C0_expanded = np.repeat(C0[:,np.newaxis], ground_pix_hom.shape[1], axis=1).reshape(3, -1)
    
    world_points = C0_expanded + l * scale
    
    # Transpose back to Nx3
    return world_points.T
