"""
Circle detector with necklace ID identification.
Based on WhyConID's CCircleDetect.cs
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .base_detector import BaseDetector


@dataclass
class Segment:
    """
    Structure containing information about detected circular pattern.
    Corresponds to SSegment in C# version.
    """
    x: float = 0.0              # Center x in image coordinates
    y: float = 0.0              # Center y in image coordinates
    angle: float = 0.0          # Orientation
    horizontal: float = 0.0     # Horizontal component
    size: int = 0               # Number of pixels
    maxy: int = 0               # Bounding box max y
    maxx: int = 0               # Bounding box max x
    miny: int = 0               # Bounding box min y
    minx: int = 0               # Bounding box min x
    mean: int = 0               # Mean brightness
    type: int = 0               # Black or white (0/1)
    roundness: float = 0.0      # Roundness test result (Eq. 2 of paper)
    bw_ratio: float = 0.0       # Ratio of white to black pixels
    round: bool = False         # Passed initial roundness test
    valid: bool = False         # Passed all tests
    m0: float = 0.0             # Covariance matrix eigenvalue
    m1: float = 0.0             # Covariance matrix eigenvalue
    v0: float = 0.0             # Covariance matrix eigenvector
    v1: float = 0.0             # Covariance matrix eigenvector
    r0: float = 0.0             # Inner vs outer ellipse ratio
    r1: float = 0.0             # Inner vs outer ellipse ratio
    ID: int = -1                # Pattern ID
    last_ID: int = -1           # Last identified ID


class CircleDetector(BaseDetector):
    """
    Main circle detection class with ID identification capability.
    Implements the WhyConID algorithm for detecting circular markers
    with necklace-style ID encoding.
    """
    
    def __init__(self, width: int, height: int, num_bots: int = 1,
                    debug: int = 0, motion_mode: bool = False):
        """
        Initialize circle detector.
        
        Args:
            width: Image width
            height: Image height
            num_bots: Number of markers to track
            debug: Debug level
            motion_mode: Enable motion-aware detection (search from last position)
        """
        super().__init__(width, height, debug)
        
        self.num_bots = num_bots
        self.identify = True
        self.local_search = False
        self.draw_inner = False
        self.draw_inner_circle = False
        
        # Detection parameters
        self.threshold = 128
        self.min_size = 50  # Moderate minimum to filter noise
        self.max_threshold = 255
        self.max_failed = 10
        self.num_failed = 0
        
        # Tolerance parameters - balanced for accuracy vs recall
        self.circular_tolerance = 0.35  # Allow some distortion
        self.circularity_tolerance = 0.1
        self.ratio_tolerance = 0.7  # Moderate area ratio tolerance
        self.center_distance_tolerance_ratio = 0.18
        self.center_distance_tolerance_abs = 18
        
        # Inner/outer area ratios
        self.outer_area_ratio = 0.0
        self.inner_area_ratio = 0.0
        self.areas_ratio = 0.0
        self.diameter_ratio = 0.0
        
        # Tracking
        self.track = motion_mode
        self.enable_corrections = True
        self.motion_mode = motion_mode
        self.search_radius = 200  # pixels to search around last position
        
        # Storage
        self.inner = Segment()
        self.outer = Segment()
        self.current_segments: List[Segment] = []
        self.last_segments: List[Segment] = []
        
        if self.debug > 0:
            print(f"CircleDetector initialized: {width}x{height}, tracking {num_bots} markers")
    
    def detect(self, image: np.ndarray) -> List[Segment]:
        """
        Detect circular markers using inner-outer ring detection.
        Implements Algorithm 2 from WhyConID paper.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            List of detected Segment objects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        binary = self._threshold_image(gray)
        
        # Find black regions (potential outer rings)
        # In motion mode, prioritize searching near last positions
        segments = self._detect_ring_pairs(binary, gray)
        
        # Identify IDs if enabled
        if self.identify:
            segments = self._identify_segments(segments, gray)
        
        # Save for next frame tracking
        self.last_segments = self.current_segments.copy()
        self.current_segments = segments
        
        if self.debug > 0:
            print(f"Detected {len(segments)} valid markers")
        
        return segments
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection.
        Simply returns grayscale image - thresholding is done during segmentation.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Grayscale image
        """
        return image
    
    def _threshold_image(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply thresholding to grayscale image.
        Uses adaptive thresholding in motion mode for better robustness.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Binary image
        """
        if self.motion_mode:
            # Use adaptive threshold for better motion tracking
            binary = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=51,  # Large block for markers
                C=5
            )
        else:
            # Fixed threshold for static images
            _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def _detect_ring_pairs(self, binary: np.ndarray,
                            gray: np.ndarray) -> List[Segment]:
        """
        Detect ring pairs (outer black ring + inner white ring).
        This is the core WhyConID algorithm with motion-aware optimization.
        
        Args:
            binary: Thresholded binary image
            gray: Original grayscale image
            
        Returns:
            List of validated Segment objects
        """
        valid_segments = []
        
        # Invert binary for black region detection
        binary_inv = cv2.bitwise_not(binary)
        
        # Find black contours (outer rings)
        contours_outer, _ = cv2.findContours(
            binary_inv,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # In motion mode, prioritize contours near last detected positions
        if self.motion_mode and len(self.last_segments) > 0:
            contours_outer = self._sort_contours_by_proximity(contours_outer)
        
        if self.debug > 1:
            print(f"Found {len(contours_outer)} potential outer rings")
        
        # Expected area ratio (from C# code)
        diameter_ratio = 50.0 / 122.0
        area_ratio_inner_outer = diameter_ratio ** 2
        expected_areas_ratio = (1 - area_ratio_inner_outer) / area_ratio_inner_outer
        
        if self.debug > 1:
            print(f"Expected areas ratio: {expected_areas_ratio:.2f}")
        
        for idx, outer_contour in enumerate(contours_outer):
            outer_area = cv2.contourArea(outer_contour)
            
            # Filter by size (avoid noise and too large regions)
            max_area = (self.width * self.height) / 4  # Max 1/4 of image
            if outer_area < self.min_size or outer_area > max_area:
                if self.debug > 2:
                    print(f"  Outer #{idx}: area {outer_area:.0f} out of range [{self.min_size}, {max_area:.0f}]")
                continue
            
            # Get outer ring properties
            outer_moments = cv2.moments(outer_contour)
            if outer_moments['m00'] == 0:
                continue
            
            outer_cx = outer_moments['m10'] / outer_moments['m00']
            outer_cy = outer_moments['m01'] / outer_moments['m00']
            
            # Check if outer ring looks circular
            outer_perimeter = cv2.arcLength(outer_contour, True)
            if outer_perimeter == 0:
                continue
            
            outer_roundness = (4 * np.pi * outer_area) / (outer_perimeter ** 2)
            
            if self.debug > 2:
                print(f"  Outer #{idx}: area={outer_area:.0f}, roundness={outer_roundness:.3f}")
            
            # Outer ring must be reasonably round
            if outer_roundness < (1.0 - self.circular_tolerance):
                if self.debug > 2:
                    print(f"    Failed roundness test")
                continue
            
            # Create a mask for the outer region
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [outer_contour], -1, 255, -1)
            
            # Find white regions inside (potential inner rings)
            inner_masked = cv2.bitwise_and(binary, mask)
            contours_inner, _ = cv2.findContours(
                inner_masked,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if self.debug > 2:
                print(f"    Found {len(contours_inner)} inner regions")
            
            # Look for a white inner region
            for inner_idx, inner_contour in enumerate(contours_inner):
                inner_area = cv2.contourArea(inner_contour)
                
                if inner_area < 5:  # Too small
                    continue
                
                # Get inner ring properties
                inner_moments = cv2.moments(inner_contour)
                if inner_moments['m00'] == 0:
                    continue
                
                inner_cx = inner_moments['m10'] / inner_moments['m00']
                inner_cy = inner_moments['m01'] / inner_moments['m00']
                
                # Check inner roundness
                inner_perimeter = cv2.arcLength(inner_contour, True)
                if inner_perimeter == 0:
                    continue
                
                inner_roundness = (4 * np.pi * inner_area) / (inner_perimeter ** 2)
                
                if self.debug > 2:
                    print(f"    Inner #{inner_idx}: area={inner_area:.0f}, roundness={inner_roundness:.3f}")
                
                if inner_roundness < (1.0 - self.circular_tolerance):
                    if self.debug > 2:
                        print(f"      Failed roundness test")
                    continue
                
                # Simplified circularity test using bounding box
                # For a circle, width ≈ height ≈ 2*radius
                x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(outer_contour)
                
                # Aspect ratio should be close to 1 for circles
                aspect_ratio = float(w_bbox) / h_bbox if h_bbox > 0 else 0
                
                if self.debug > 2:
                    print(f"      BBox: {w_bbox}x{h_bbox}, aspect={aspect_ratio:.3f}")
                
                # Aspect ratio should be close to 1 for circles (allow some distortion for motion blur)
                if aspect_ratio < 0.6 or aspect_ratio > 1.7:
                    if self.debug > 2:
                        print(f"      Failed aspect ratio test")
                    continue
                
                # Check area ratio (black/white ratio)
                actual_ratio = outer_area / inner_area
                ratio_min = expected_areas_ratio - self.ratio_tolerance
                ratio_max = expected_areas_ratio + self.ratio_tolerance
                
                if self.debug > 2:
                    print(f"      Area ratio: {actual_ratio:.2f} (expected: {expected_areas_ratio:.2f} ± {self.ratio_tolerance})")
                
                if not (ratio_min < actual_ratio < ratio_max):
                    if self.debug > 2:
                        print(f"      Failed area ratio test")
                    continue
                
                # Check concentricity (centers should be close)
                center_distance = np.sqrt((outer_cx - inner_cx)**2 + (outer_cy - inner_cy)**2)
                max_allowed_distance = (self.center_distance_tolerance_abs + 
                                      self.center_distance_tolerance_ratio * max(w_bbox, h_bbox))
                
                if self.debug > 2:
                    print(f"      Center distance: {center_distance:.1f} (max: {max_allowed_distance:.1f})")
                
                if center_distance > max_allowed_distance:
                    if self.debug > 2:
                        print(f"      Failed concentricity test")
                    continue
                
                # Estimate principal axes from moments (for segment info)
                covar = np.array([
                    [outer_moments['mu20'], outer_moments['mu11']],
                    [outer_moments['mu11'], outer_moments['mu02']]
                ]) / outer_moments['m00']
                
                eigenvalues = np.linalg.eigvalsh(covar)
                m0, m1 = float(np.sqrt(abs(eigenvalues[1]))), float(np.sqrt(abs(eigenvalues[0])))
                
                # All tests passed! Create valid segment
                seg = Segment(
                    x=outer_cx,
                    y=outer_cy,
                    size=int(outer_area),
                    minx=x_bbox,
                    miny=y_bbox,
                    maxx=x_bbox + w_bbox,
                    maxy=y_bbox + h_bbox,
                    roundness=outer_roundness,
                    bw_ratio=inner_area / outer_area,
                    round=True,
                    valid=True,
                    m0=m0,
                    m1=m1
                )
                
                if self.debug > 1:
                    print(f"  ✓ VALID marker found at ({outer_cx:.1f}, {outer_cy:.1f})")
                
                valid_segments.append(seg)
                break  # Found valid inner ring, move to next outer ring
        
        return valid_segments
    
    def _sort_contours_by_proximity(self, contours: List) -> List:
        """
        Sort contours by proximity to last detected positions.
        This implements C#'s motion-aware detection optimization.
        
        Args:
            contours: List of contours
            
        Returns:
            Sorted list of contours (closest first)
        """
        if not contours or not self.last_segments:
            return contours
        
        # Calculate distance score for each contour
        scored_contours = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
            # Find minimum distance to any last position
            min_dist = float('inf')
            for last_seg in self.last_segments:
                if last_seg.valid:
                    dist = np.sqrt((cx - last_seg.x)**2 + (cy - last_seg.y)**2)
                    if dist < min_dist:
                        min_dist = dist
            
            # Prioritize contours within search radius
            if min_dist < self.search_radius:
                scored_contours.append((min_dist, contour))
            else:
                # Add far contours at the end with penalty
                scored_contours.append((min_dist + 10000, contour))
        
        # Sort by distance (closest first)
        scored_contours.sort(key=lambda x: x[0])
        
        return [c for _, c in scored_contours]
    
    
    def _filter_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Additional filtering (already done in _detect_ring_pairs).
        
        Args:
            segments: List of segments
            
        Returns:
            Filtered list of segments
        """
        return segments  # Already validated in ring detection
    
    
    def _identify_segments(self, segments: List[Segment],
                            image: np.ndarray) -> List[Segment]:
        """
        Identify marker IDs for segments.
        
        Args:
            segments: List of segments
            image: Original grayscale image
            
        Returns:
            Segments with identified IDs
        """
        # TODO: Implement necklace ID identification
        # This will be integrated with CNecklace class
        
        for i, seg in enumerate(segments):
            seg.ID = i  # Temporary: assign sequential IDs
        
        return segments
    
    def reset(self):
        """Reset detector state."""
        self.current_segments = []
        self.last_segments = []
        self.num_failed = 0
        self.last_track_ok = False
        
        if self.debug > 0:
            print("CircleDetector reset")
