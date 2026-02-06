"""
Base detector class providing common functionality for all detectors.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for all marker detectors.
    """
    
    def __init__(self, width: int, height: int, debug: int = 0):
        """
        Initialize base detector.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            debug: Debug level (0=off, 1=minimal, 2=verbose)
        """
        self.width = width
        self.height = height
        self.debug = debug
        self.draw = False
        self.draw_all = False
        self.last_track_ok = False
        
    @abstractmethod
    def detect(self, image: np.ndarray) -> List:
        """
        Detect markers in the given image.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            List of detected markers
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset detector state."""
        pass
    
    def set_debug(self, level: int):
        """Set debug level."""
        self.debug = level
        
    def enable_drawing(self, enabled: bool = True):
        """Enable/disable drawing of detection results."""
        self.draw = enabled
