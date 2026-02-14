"""
Depth estimation from marker size using monocular camera.
Based on pinhole camera model.
"""

import numpy as np
from typing import Optional, Tuple
from detectors.circle_detect import Segment


class DepthEstimator:
    """
    Estimate depth (Z-axis distance) from marker size in image.
    
    Uses pinhole camera model:
    Z = (f * D_real) / D_pixel
    
    where:
    - Z: depth (distance to camera)
    - f: focal length in pixels
    - D_real: real-world diameter of marker
    - D_pixel: pixel diameter in image
    """
    
    def __init__(self,
                 focal_length_px: Optional[float] = None,
                 focal_length_mm: Optional[float] = None,
                 sensor_width_mm: Optional[float] = None,
                 image_width_px: Optional[int] = None,
                 fov_horizontal_deg: Optional[float] = None,
                 marker_diameter_mm: float = 50.0,
                 camera_transform = None):
        """
        Initialize depth estimator.
        
        Provide ONE of the following combinations for focal length:
        1. focal_length_px (directly in pixels)
        2. focal_length_mm + sensor_width_mm + image_width_px (from camera specs)
        3. fov_horizontal_deg + image_width_px (from FOV)
        
        Args:
            focal_length_px: Focal length in pixels (pre-calculated)
            focal_length_mm: Focal length in millimeters (camera spec)
            sensor_width_mm: Sensor width in millimeters (camera spec)
            image_width_px: Image width in pixels
            fov_horizontal_deg: Horizontal field of view in degrees
            marker_diameter_mm: Real-world outer diameter of marker (default: 50mm)
            camera_transform: CameraTransform instance for world coordinate conversion (optional)
        """
        self.marker_diameter_mm = marker_diameter_mm
        self.camera_transform = camera_transform
        
        # Calculate focal length in pixels
        if focal_length_px is not None:
            self.focal_length_px = focal_length_px
        elif focal_length_mm is not None and sensor_width_mm is not None and image_width_px is not None:
            # f_px = f_mm * (image_width_px / sensor_width_mm)
            self.focal_length_px = focal_length_mm * (image_width_px / sensor_width_mm)
        elif fov_horizontal_deg is not None and image_width_px is not None:
            # f_px = image_width / (2 * tan(FOV/2))
            fov_rad = np.deg2rad(fov_horizontal_deg)
            self.focal_length_px = image_width_px / (2.0 * np.tan(fov_rad / 2.0))
        else:
            # Default estimate for typical webcam (FOV ~60°, 640px width)
            print("Warning: No focal length provided, using default estimate (FOV=60°, width=640px)")
            self.focal_length_px = 640 / (2.0 * np.tan(np.deg2rad(60) / 2.0))
        
        print(f"DepthEstimator initialized: focal_length={self.focal_length_px:.1f}px, marker_diameter={marker_diameter_mm}mm")
    
    def estimate_depth(self, segment: Segment) -> Optional[float]:
        """
        Estimate depth from segment.
        
        Args:
            segment: Detected segment with bounding box
            
        Returns:
            Depth in meters, or None if estimation fails
        """
        # Calculate pixel diameter from bounding box
        pixel_diameter = self._get_pixel_diameter(segment)
        
        if pixel_diameter < 1.0:
            return None
        
        # Apply pinhole camera model
        # Z = (f * D_real) / D_pixel
        depth_mm = (self.focal_length_px * self.marker_diameter_mm) / pixel_diameter
        depth_m = depth_mm / 1000.0
        
        return depth_m
    
    def estimate_3d_position(self, 
                            segment: Segment, 
                            image_width: int, 
                            image_height: int) -> Optional[Tuple[float, float, float]]:
        """
        Estimate 3D position (X, Y, Z) in camera coordinate system.
        
        Camera coordinates:
        - X: right
        - Y: down
        - Z: forward (depth)
        - Origin: camera center
        
        Args:
            segment: Detected segment
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            (X, Y, Z) in meters, or None if estimation fails
        """
        # Get depth
        Z = self.estimate_depth(segment)
        if Z is None:
            return None
        
        # Image center (principal point, assuming centered)
        cx = image_width / 2.0
        cy = image_height / 2.0
        
        # Pixel coordinates relative to center
        u = segment.x - cx
        v = segment.y - cy
        
        # Back-project to 3D
        # X = (u * Z) / f
        # Y = (v * Z) / f
        X = (u * Z) / self.focal_length_px
        Y = (v * Z) / self.focal_length_px
        
        return (X, Y, Z)
    
    def estimate_world_position(self,
                               segment: Segment,
                               image_width: int,
                               image_height: int) -> Optional[Tuple[float, float, float]]:
        """
        Estimate 3D position in world coordinate system.
        
        Requires camera_transform to be set.
        
        World coordinates:
        - X: forward
        - Y: left  
        - Z: up
        - Origin: defined by camera_transform
        
        Args:
            segment: Detected segment
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            (X, Y, Z) in world frame (meters), or None if estimation fails
        """
        if self.camera_transform is None:
            return None
        
        # Get camera coordinates
        pos_cam = self.estimate_3d_position(segment, image_width, image_height)
        if pos_cam is None:
            return None
        
        # Transform to world coordinates
        pos_world = self.camera_transform.camera_to_world(pos_cam)
        
        return pos_world
    
    def estimate_ground_position(self,
                                segment: Segment,
                                image_width: int,
                                image_height: int,
                                ground_z: float = 0.0) -> Optional[Tuple[float, float, float]]:
        """
        Estimate position where marker ray intersects ground plane.
        
        Useful when markers are on ground or at known height.
        
        Args:
            segment: Detected segment
            image_width: Image width in pixels
            image_height: Image height in pixels
            ground_z: Ground plane Z coordinate in world frame (default: 0)
            
        Returns:
            Ground intersection point (X, Y, Z) in world frame, or None
        """
        if self.camera_transform is None:
            return None
        
        # Get camera coordinates
        pos_cam = self.estimate_3d_position(segment, image_width, image_height)
        if pos_cam is None:
            return None
        
        # Project onto ground
        ground_pos = self.camera_transform.get_ground_position(pos_cam, ground_z)
        
        return ground_pos
    
    def _get_pixel_diameter(self, segment: Segment) -> float:
        """
        Extract pixel diameter from segment.
        
        Uses diagonal of bounding box as approximation.
        For circular markers, diagonal ≈ sqrt(2) * diameter,
        so we apply correction factor.
        
        Args:
            segment: Detected segment
            
        Returns:
            Estimated pixel diameter
        """
        # Bounding box dimensions
        width_px = segment.maxx - segment.minx
        height_px = segment.maxy - segment.miny
        
        # Use average of width/height for better robustness
        # (handles slight ellipse distortion from camera angle)
        diameter_px = (width_px + height_px) / 2.0
        
        return diameter_px
    
    def update_marker_diameter(self, diameter_mm: float):
        """Update marker real-world diameter."""
        self.marker_diameter_mm = diameter_mm
        print(f"Updated marker diameter to {diameter_mm}mm")
