"""
Image preprocessing utilities for marker detection.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class ImageProcessor:
    """
    Image preprocessing and enhancement for marker detection.
    """
    
    @staticmethod
    def preprocess(image: np.ndarray,
                    enhance_contrast: bool = True,
                    denoise: bool = True) -> np.ndarray:
        """
        Preprocess image for detection.
        
        Args:
            image: Input image (grayscale or BGR)
            enhance_contrast: Apply CLAHE contrast enhancement
            denoise: Apply denoising
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Enhance contrast
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        return gray
    
    @staticmethod
    def adaptive_threshold(image: np.ndarray,
                            block_size: int = 11,
                            C: int = 2) -> np.ndarray:
        """
        Apply adaptive thresholding.
        
        Args:
            image: Grayscale input image
            block_size: Size of pixel neighborhood
            C: Constant subtracted from mean
            
        Returns:
            Binary image
        """
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )
    
    @staticmethod
    def morphological_cleanup(binary: np.ndarray,
                                operation: str = 'close',
                                kernel_size: int = 3) -> np.ndarray:
        """
        Apply morphological operations to clean binary image.
        
        Args:
            binary: Binary input image
            operation: 'open', 'close', 'erode', or 'dilate'
            kernel_size: Size of structuring element
            
        Returns:
            Processed binary image
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        if operation == 'open':
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operation == 'erode':
            return cv2.erode(binary, kernel)
        elif operation == 'dilate':
            return cv2.dilate(binary, kernel)
        else:
            return binary
    
    @staticmethod
    def detect_edges(image: np.ndarray,
                    low_threshold: Optional[int] = None,
                    high_threshold: Optional[int] = None) -> np.ndarray:
        """
        Detect edges using Canny algorithm with automatic thresholds.
        
        Args:
            image: Grayscale input image
            low_threshold: Low threshold (auto if None)
            high_threshold: High threshold (auto if None)
            
        Returns:
            Edge image
        """
        if low_threshold is None or high_threshold is None:
            # Automatic threshold selection
            median = np.median(image)
            low_threshold = int(max(0, 0.66 * median))
            high_threshold = int(min(255, 1.33 * median))
        
        return cv2.Canny(image, low_threshold, high_threshold)
