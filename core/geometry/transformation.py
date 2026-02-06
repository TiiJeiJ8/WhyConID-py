"""
Coordinate transformation utilities.
Based on CTransformation.cs

Handles image-to-world coordinate transformations, camera calibration,
and perspective corrections.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class TransformType(Enum):
    """Type of coordinate transformation."""
    NONE = 0        # Camera-centric
    TRANSFORM_2D = 1    # 3D->2D homography
    TRANSFORM_3D = 2    # 3D user-defined
    TRANSFORM_4D = 3    # Full 4x3 matrix


@dataclass
class TrackedObject:
    """
    Tracked object in metric space.
    Corresponds to STrackedObject in C# version.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    d: float = 0.0          # Distance from camera
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    roundness: float = 0.0
    bw_ratio: float = 0.0
    error: float = 0.0
    est_error: float = 0.0
    ID: int = -1


class CoordinateTransform:
    """
    Coordinate transformation between image and world coordinates.
    
    Handles camera calibration, distortion correction, and various
    transformation models (2D homography, 3D transformation).
    """
    
    def __init__(self, width: int, height: int,
                    marker_diameter: float,
                    transform_type: TransformType = TransformType.NONE):
        """
        Initialize coordinate transformer.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            marker_diameter: Physical diameter of markers (in meters or chosen unit)
            transform_type: Type of transformation to use
        """
        self.width = width
        self.height = height
        self.marker_diameter = marker_diameter
        self.transform_type = transform_type
        
        # Camera intrinsic parameters
        self.fc = np.array([1.0, 1.0])  # Focal length [fx, fy]
        self.cc = np.array([width/2.0, height/2.0])  # Principal point
        self.kc = np.zeros(5)  # Distortion coefficients [k1, k2, p1, p2, k3]
        
        # Homography matrix (for 2D transform)
        self.homography = np.eye(3)
        
        # Calibration state
        self.is_calibrated = False
        
    def set_camera_parameters(self,
                                focal_length: Tuple[float, float],
                                principal_point: Tuple[float, float],
                                distortion: Optional[np.ndarray] = None):
        """
        Set camera intrinsic parameters.
        
        Args:
            focal_length: (fx, fy) focal lengths in pixels
            principal_point: (cx, cy) principal point
            distortion: Distortion coefficients [k1, k2, p1, p2, k3]
        """
        self.fc = np.array(focal_length)
        self.cc = np.array(principal_point)
        
        if distortion is not None:
            self.kc = distortion
        
        self.is_calibrated = True
    
    def undistort_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Remove lens distortion from a point.
        
        Args:
            point: (x, y) distorted point coordinates
            
        Returns:
            (x, y) undistorted coordinates
        """
        if not self.is_calibrated or np.all(self.kc == 0):
            return point
        
        x, y = point
        
        # Normalize coordinates
        x_n = (x - self.cc[0]) / self.fc[0]
        y_n = (y - self.cc[1]) / self.fc[1]
        
        # Radial distance
        r2 = x_n**2 + y_n**2
        r4 = r2**2
        r6 = r2**3
        
        # Radial distortion
        k1, k2, p1, p2, k3 = self.kc if len(self.kc) >= 5 else (*self.kc, 0, 0, 0)
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        
        # Tangential distortion
        dx = 2*p1*x_n*y_n + p2*(r2 + 2*x_n**2)
        dy = p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n
        
        # Apply correction
        x_u = x_n * radial + dx
        y_u = y_n * radial + dy
        
        # Denormalize
        x_out = x_u * self.fc[0] + self.cc[0]
        y_out = y_u * self.fc[1] + self.cc[1]
        
        return (x_out, y_out)
    
    def image_to_world(self,
                        image_point: Tuple[float, float],
                        z: float = 0.0) -> Tuple[float, float, float]:
        """
        Transform image coordinates to world coordinates.
        
        Args:
            image_point: (x, y) in image coordinates
            z: World Z coordinate (height), default 0 for planar
            
        Returns:
            (x, y, z) in world coordinates
        """
        if self.transform_type == TransformType.NONE:
            # Camera-centric: just undistort
            x, y = self.undistort_point(image_point)
            return (x, y, z)
        
        elif self.transform_type == TransformType.TRANSFORM_2D:
            # Apply homography
            x, y = image_point
            point_h = np.array([x, y, 1.0])
            world_h = self.homography @ point_h
            
            if world_h[2] != 0:
                wx = world_h[0] / world_h[2]
                wy = world_h[1] / world_h[2]
            else:
                wx, wy = x, y
            
            return (wx, wy, z)
        
        else:
            # TODO: Implement 3D/4D transformations
            return (*image_point, z)
    
    def world_to_image(self,
                        world_point: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Transform world coordinates to image coordinates.
        
        Args:
            world_point: (x, y, z) in world coordinates
            
        Returns:
            (x, y) in image coordinates
        """
        if self.transform_type == TransformType.TRANSFORM_2D:
            # Apply inverse homography
            x, y, z = world_point
            point_h = np.array([x, y, 1.0])
            
            try:
                inv_h = np.linalg.inv(self.homography)
                image_h = inv_h @ point_h
                
                if image_h[2] != 0:
                    ix = image_h[0] / image_h[2]
                    iy = image_h[1] / image_h[2]
                else:
                    ix, iy = x, y
                
                return (ix, iy)
            except np.linalg.LinAlgError:
                return (x, y)
        
        else:
            return (world_point[0], world_point[1])
    
    def estimate_homography(self,
                            image_points: List[Tuple[float, float]],
                            world_points: List[Tuple[float, float]]) -> bool:
        """
        Estimate homography from point correspondences.
        
        Args:
            image_points: List of (x, y) in image coordinates
            world_points: List of (x, y) in world coordinates
            
        Returns:
            True if successful
        """
        if len(image_points) < 4 or len(image_points) != len(world_points):
            return False
        
        try:
            import cv2
            
            src_pts = np.array(image_points, dtype=np.float32)
            dst_pts = np.array(world_points, dtype=np.float32)
            
            self.homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            
            return self.homography is not None
        except:
            return False
