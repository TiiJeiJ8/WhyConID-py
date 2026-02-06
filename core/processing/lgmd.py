"""
LGMD (Lobula Giant Movement Detector) processing.
Bio-inspired motion detection and temporal filtering.
Based on LGMDProcessing.cs
"""

import numpy as np
import cv2
from typing import Optional


class LGMDProcessor:
    """
    LGMD-based motion detection and temporal processing.
    
    Implements bio-inspired vision algorithms for detecting motion
    and temporal changes in image sequences.
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize LGMD processor.
        
        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height
        
        # Previous frame storage
        self.prev_frame: Optional[np.ndarray] = None
        
        # Temporal filter parameters
        self.decay_factor = 0.9
        self.threshold = 20
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image with LGMD-style temporal filtering.
        
        Args:
            image: Current frame (grayscale)
            
        Returns:
            Processed image highlighting temporal changes
        """
        if self.prev_frame is None:
            self.prev_frame = image.copy().astype(np.float32)
            return np.zeros_like(image)
        
        # Convert to float
        current = image.astype(np.float32)
        
        # Temporal difference
        diff = cv2.absdiff(current, self.prev_frame)
        
        # Threshold to detect significant changes
        _, motion_mask = cv2.threshold(
            diff.astype(np.uint8),
            self.threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        # Update previous frame with decay
        self.prev_frame = (self.decay_factor * self.prev_frame + 
                          (1 - self.decay_factor) * current)
        
        return motion_mask
    
    def reset(self):
        """Reset processor state."""
        self.prev_frame = None
