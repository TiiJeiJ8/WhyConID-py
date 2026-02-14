"""
Camera coordinate transformation.
Convert between camera coordinates and world coordinates using extrinsic parameters.
"""

import numpy as np
from typing import Tuple, Optional


class CameraTransform:
    """
    Camera extrinsic parameters and coordinate transformation.
    
    Coordinate systems:
    - Camera: X=right, Y=down, Z=forward
    - World: X=forward, Y=left, Z=up (right-hand rule)
    """
    
    def __init__(self,
                 position: Tuple[float, float, float] = (0.0, 0.0, 1.5),
                 rotation: Tuple[float, float, float] = (0.0, 45.0, 0.0)):
        """
        Initialize camera transform.
        
        Args:
            position: Camera position in world frame (X, Y, Z) in meters
                     Default: (0, 0, 1.5) = 1.5m above origin
            rotation: Camera rotation in degrees (roll, pitch, yaw)
                     roll: rotation around X (forward) axis
                     pitch: rotation around Y (left) axis - positive = looking down
                     yaw: rotation around Z (up) axis - positive = turning left
                     Default: (0, 45, 0) = looking down 45 degrees
        """
        self.position = np.array(position, dtype=float)
        self.rotation_deg = np.array(rotation, dtype=float)
        self.rotation_rad = np.deg2rad(self.rotation_deg)
        
        # Build rotation matrix (world -> camera)
        self.R_world_to_cam = self._build_rotation_matrix()
        # Inverse (camera -> world)
        self.R_cam_to_world = self.R_world_to_cam.T
        
        print(f"CameraTransform initialized:")
        print(f"  Position (world): {self.position}")
        print(f"  Rotation (deg): roll={rotation[0]}, pitch={rotation[1]}, yaw={rotation[2]}")
    
    def _build_rotation_matrix(self) -> np.ndarray:
        """
        Build rotation matrix from Euler angles (ZYX order).
        
        Returns:
            3x3 rotation matrix (world -> camera)
        """
        roll, pitch, yaw = self.rotation_rad
        
        # Individual rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation (ZYX order)
        R = Rz @ Ry @ Rx
        
        return R
    
    def camera_to_world(self, point_cam: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Transform point from camera coordinates to world coordinates.
        
        Args:
            point_cam: Point in camera frame (X, Y, Z)
                      Camera frame: X=right, Y=down, Z=forward
        
        Returns:
            Point in world frame (X, Y, Z)
            World frame: X=forward, Y=left, Z=up
        """
        point_cam = np.array(point_cam)
        
        # Rotate from camera to world
        point_world_relative = self.R_cam_to_world @ point_cam
        
        # Translate by camera position
        point_world = point_world_relative + self.position
        
        return tuple(point_world)
    
    def world_to_camera(self, point_world: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Transform point from world coordinates to camera coordinates.
        
        Args:
            point_world: Point in world frame (X, Y, Z)
        
        Returns:
            Point in camera frame (X, Y, Z)
        """
        point_world = np.array(point_world)
        
        # Translate to camera origin
        point_relative = point_world - self.position
        
        # Rotate to camera frame
        point_cam = self.R_world_to_cam @ point_relative
        
        return tuple(point_cam)
    
    def get_ground_position(self, 
                           point_cam: Tuple[float, float, float],
                           ground_z: float = 0.0) -> Optional[Tuple[float, float, float]]:
        """
        Project camera point onto ground plane.
        
        Args:
            point_cam: Point in camera frame
            ground_z: Z coordinate of ground plane in world frame (default: 0)
        
        Returns:
            Ground intersection point in world frame, or None if no intersection
        """
        # Transform to world
        point_world = self.camera_to_world(point_cam)
        
        # Ray from camera to point
        ray_origin = self.position
        ray_direction = point_world - ray_origin
        
        # Normalize
        ray_length = np.linalg.norm(ray_direction)
        if ray_length < 1e-6:
            return None
        ray_direction = ray_direction / ray_length
        
        # Intersect with ground plane Z=ground_z
        # Point = origin + t * direction
        # Z = origin_z + t * direction_z = ground_z
        # t = (ground_z - origin_z) / direction_z
        
        if abs(ray_direction[2]) < 1e-6:  # Ray parallel to ground
            return None
        
        t = (ground_z - ray_origin[2]) / ray_direction[2]
        
        if t < 0:  # Intersection behind camera
            return None
        
        intersection = ray_origin + t * ray_direction
        
        return tuple(intersection)
    
    def update_position(self, position: Tuple[float, float, float]):
        """Update camera position."""
        self.position = np.array(position, dtype=float)
        print(f"Camera position updated to: {self.position}")
    
    def update_rotation(self, rotation: Tuple[float, float, float]):
        """Update camera rotation."""
        self.rotation_deg = np.array(rotation, dtype=float)
        self.rotation_rad = np.deg2rad(self.rotation_deg)
        self.R_world_to_cam = self._build_rotation_matrix()
        self.R_cam_to_world = self.R_world_to_cam.T
        print(f"Camera rotation updated to: {self.rotation_deg}")
