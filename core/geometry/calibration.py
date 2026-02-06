"""
Camera calibration utilities.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class CameraCalibrator:
    """
    Camera calibration for intrinsic and extrinsic parameters.
    """
    
    def __init__(self):
        """Initialize calibrator."""
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: List[np.ndarray] = []
        self.tvecs: List[np.ndarray] = []
        
    def calibrate_from_chessboard(self,
                                    images: List[np.ndarray],
                                    pattern_size: Tuple[int, int],
                                    square_size: float) -> bool:
        """
        Calibrate camera using chessboard pattern.
        
        Args:
            images: List of calibration images
            pattern_size: (rows, cols) of inner corners
            square_size: Size of chessboard squares in world units
            
        Returns:
            True if calibration successful
        """
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0],
                                0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        obj_points = []  # 3D points in real world
        img_points = []  # 2D points in image plane
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                obj_points.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_refined)
        
        if len(obj_points) == 0:
            return False
        
        # Calibrate
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        
        return ret
    
    def get_parameters(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get calibration parameters.
        
        Returns:
            (camera_matrix, distortion_coefficients)
        """
        return self.camera_matrix, self.dist_coeffs
