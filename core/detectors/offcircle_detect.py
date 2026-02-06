"""
Off-axis and ellipse-based circle detector.
Handles perspective distortion and eccentric circles.
Based on OffCircleDetect.cs
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from .base_detector import BaseDetector
from .circle_detect import Segment


class OffCircleDetector(BaseDetector):
    """
    Detector for circles that appear as ellipses due to perspective projection
    or off-axis viewing angles.
    """
    
    def __init__(self, width: int, height: int, debug: int = 0):
        """
        Initialize off-circle detector.
        
        Args:
            width: Image width
            height: Image height
            debug: Debug level
        """
        super().__init__(width, height, debug)
        
        # Ellipse fitting parameters
        self.min_points_for_ellipse = 5
        self.eccentricity_threshold = 0.9
        
    def detect(self, image: np.ndarray) -> List[Segment]:
        """
        Detect elliptical markers in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected segments
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Fit ellipses
        segments = self._fit_ellipses(contours)
        
        return segments
    
    def _fit_ellipses(self, contours: List) -> List[Segment]:
        """
        Fit ellipses to contours and create segments.
        
        Args:
            contours: List of contours
            
        Returns:
            List of segments with ellipse parameters
        """
        segments = []
        
        for contour in contours:
            # Need at least 5 points to fit ellipse
            if len(contour) < self.min_points_for_ellipse:
                continue
            
            try:
                # Fit ellipse
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (w, h), angle = ellipse
                
                # Calculate eccentricity
                a = max(w, h) / 2.0
                b = min(w, h) / 2.0
                if a > 0:
                    eccentricity = np.sqrt(1 - (b**2 / a**2))
                else:
                    continue
                
                # Skip near-linear shapes
                if eccentricity > self.eccentricity_threshold:
                    continue
                
                # Create segment
                seg = Segment(
                    x=cx,
                    y=cy,
                    angle=angle,
                    m0=a,  # Semi-major axis
                    m1=b,  # Semi-minor axis
                    size=len(contour),
                    valid=True
                )
                
                segments.append(seg)
                
            except cv2.error:
                continue
        
        return segments
    
    def reset(self):
        """Reset detector state."""
        pass
