"""
Visualization utilities for WhyConID detection results.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from datetime import datetime
from detectors.circle_detect import Segment


class DetectionVisualizer:
    """
    Visualizer for detection results with annotated images.
    """
    
    def __init__(self, output_dir: str = "output", use_color_trajectory: bool = False):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save output images
            use_color_trajectory: Use different colors for each track
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_color_trajectory = use_color_trajectory
        
        # Colors (BGR format)
        self.color_center = (0, 255, 0)      # Green for center
        self.color_bbox = (255, 0, 0)        # Blue for bounding box
        self.color_id = (0, 255, 255)        # Yellow for ID text
        self.color_info = (255, 255, 255)    # White for info text
        self.color_trajectory = (0, 165, 255) # Orange for trajectory (default)
        self.color_prediction = (255, 0, 255) # Magenta for prediction
        
        # Predefined colors for different tracks (vivid colors, BGR format)
        self.track_colors = [
            (0, 165, 255),   # Orange
            (255, 0, 255),   # Magenta
            (0, 255, 0),     # Green
            (255, 255, 0),   # Cyan
            (0, 255, 255),   # Yellow
            (255, 0, 127),   # Purple
            (0, 127, 255),   # Coral
            (127, 255, 0),   # Lime
            (255, 127, 0),   # Sky blue
            (127, 0, 255),   # Pink
        ]
        
    def draw_segments(self,
                        image: np.ndarray,
                        segments: List[Segment],
                        show_bbox: bool = True,
                        show_center: bool = True,
                        show_id: bool = True,
                        show_info: bool = True,
                        trajectories: Optional[Dict] = None,
                        track_ids: Optional[Dict] = None,
                        predictions: Optional[Dict] = None,
                        show_prediction: bool = False,
                        previous_predictions: Optional[Dict] = None,
                        show_prediction_error: bool = False) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image
            segments: List of detected segments
            show_bbox: Draw bounding boxes
            show_center: Draw center points
            show_id: Draw marker IDs
            show_info: Draw statistics
            trajectories: Dict of track_id -> list of (x, y) positions
            track_ids: Dict mapping segment index to track_id
            predictions: Dict of track_id -> list of future (x, y) positions
            show_prediction: Whether to show predictions
            previous_predictions: Dict of track_id -> list of previously predicted positions
            show_prediction_error: Whether to show prediction error visualization
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw trajectories first (so they appear behind markers)
        if trajectories:
            for track_id, trajectory in trajectories.items():
                if len(trajectory) > 1:
                    # Choose color based on track ID
                    if self.use_color_trajectory:
                        color = self.track_colors[track_id % len(self.track_colors)]
                    else:
                        color = self.color_trajectory
                    
                    # Draw trajectory line
                    points = np.array(trajectory, dtype=np.int32)
                    cv2.polylines(
                        annotated,
                        [points],
                        False,
                        color,
                        2
                    )
                    
                    # Draw trajectory points
                    for point in trajectory[-10:]:  # Last 10 points
                        cv2.circle(
                            annotated,
                            (int(point[0]), int(point[1])),
                            2,
                            color,
                            -1
                        )
        
        # Draw predictions (after trajectories, before markers)
        if show_prediction and predictions and trajectories:
            # Create overlay for semi-transparent predictions
            overlay = annotated.copy()
            
            for track_id, future_positions in predictions.items():
                if not future_positions or track_id not in trajectories:
                    continue
                
                # Get track color (brighter version for prediction)
                if self.use_color_trajectory:
                    base_color = self.track_colors[track_id % len(self.track_colors)]
                else:
                    base_color = self.color_prediction
                
                # Make prediction color brighter (increase all channels by 40%)
                pred_color = tuple(min(int(c * 1.4), 255) for c in base_color)
                
                # Get current position (last trajectory point)
                trajectory = trajectories[track_id]
                if len(trajectory) == 0:
                    continue
                current_pos = trajectory[-1]
                
                # 方案A: Draw arrow from current to first predicted position
                if len(future_positions) > 0:
                    first_pred = future_positions[0]
                    cv2.arrowedLine(
                        overlay,
                        (int(current_pos[0]), int(current_pos[1])),
                        (int(first_pred[0]), int(first_pred[1])),
                        pred_color,
                        3,
                        tipLength=0.25
                    )
                
                # 方案B: Draw multi-step prediction trajectory (dashed line)
                if len(future_positions) > 1:
                    # Create list of all prediction points
                    pred_points = np.array(future_positions, dtype=np.int32)
                    
                    # Draw dashed line by manually drawing segments
                    dash_length = 10
                    gap_length = 5
                    
                    for i in range(len(pred_points) - 1):
                        pt1 = pred_points[i]
                        pt2 = pred_points[i + 1]
                        
                        # Calculate segment length
                        dx = pt2[0] - pt1[0]
                        dy = pt2[1] - pt1[1]
                        segment_length = np.sqrt(dx**2 + dy**2)
                        
                        if segment_length < 1:
                            continue
                        
                        # Normalize direction
                        dx /= segment_length
                        dy /= segment_length
                        
                        # Draw dashed line
                        current_length = 0
                        is_drawing = True
                        while current_length < segment_length:
                            if is_drawing:
                                # Draw dash
                                start_x = int(pt1[0] + dx * current_length)
                                start_y = int(pt1[1] + dy * current_length)
                                end_length = min(current_length + dash_length, segment_length)
                                end_x = int(pt1[0] + dx * end_length)
                                end_y = int(pt1[1] + dy * end_length)
                                
                                cv2.line(
                                    overlay,
                                    (start_x, start_y),
                                    (end_x, end_y),
                                    pred_color,
                                    2,
                                    cv2.LINE_AA
                                )
                                current_length += dash_length
                                is_drawing = False
                            else:
                                # Gap
                                current_length += gap_length
                                is_drawing = True
                    
                    # Draw hollow circles at prediction points for better visibility
                    for point in pred_points:
                        cv2.circle(
                            overlay,
                            (int(point[0]), int(point[1])),
                            4,
                            pred_color,
                            2  # Hollow circle
                        )
            
            # Blend overlay with original (60% opacity for predictions)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # 方案C: Draw prediction error (compare predicted vs actual)
        if show_prediction_error and previous_predictions and trajectories:
            for track_id, prev_preds in previous_predictions.items():
                if track_id not in trajectories or not prev_preds:
                    continue
                
                trajectory = trajectories[track_id]
                if len(trajectory) < 2:
                    continue
                
                # Get actual current position
                actual_pos = trajectory[-1]
                
                # The first prediction from previous frame should match current position
                if len(prev_preds) > 0:
                    predicted_pos = prev_preds[0]
                    
                    # Calculate prediction error
                    error_dx = actual_pos[0] - predicted_pos[0]
                    error_dy = actual_pos[1] - predicted_pos[1]
                    error_distance = np.sqrt(error_dx**2 + error_dy**2)
                    
                    # Draw error line (from predicted to actual)
                    cv2.line(
                        annotated,
                        (int(predicted_pos[0]), int(predicted_pos[1])),
                        (int(actual_pos[0]), int(actual_pos[1])),
                        (0, 0, 255),  # Red for error
                        2,
                        cv2.LINE_AA
                    )
                    
                    # Draw predicted position as X mark
                    cv2.drawMarker(
                        annotated,
                        (int(predicted_pos[0]), int(predicted_pos[1])),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        10,
                        2
                    )
                    
                    # Draw error distance text
                    text_pos = (int(predicted_pos[0]) + 10, int(predicted_pos[1]) - 10)
                    cv2.putText(
                        annotated,
                        f"Error: {error_distance:.1f}px",
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )

        
        for i, seg in enumerate(segments):
            # Draw bounding box
            if show_bbox:
                cv2.rectangle(
                    annotated,
                    (seg.minx, seg.miny),
                    (seg.maxx, seg.maxy),
                    self.color_bbox,
                    2
                )
            
            # Draw center point
            if show_center:
                cv2.circle(
                    annotated,
                    (int(seg.x), int(seg.y)),
                    6,
                    self.color_center,
                    -1
                )
                # Draw a small cross
                cv2.drawMarker(
                    annotated,
                    (int(seg.x), int(seg.y)),
                    self.color_center,
                    cv2.MARKER_CROSS,
                    12,
                    2
                )
            
            # Draw ID and details
            if show_id:
                # Use track ID if available, otherwise use segment ID
                display_id = seg.ID
                id_suffix = ""
                if track_ids and i in track_ids:
                    display_id = f"T{track_ids[i]}"
                elif seg.ID < 0:
                    # No valid ID, use index
                    display_id = f"#{i}"
                
                # Main ID label
                label = f"ID: {display_id}{id_suffix}"
                label_pos = (int(seg.x) + 15, int(seg.y) - 15)
                
                # Draw text background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    2
                )
                cv2.rectangle(
                    annotated,
                    (label_pos[0] - 2, label_pos[1] - label_h - 2),
                    (label_pos[0] + label_w + 2, label_pos[1] + 2),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.color_id,
                    2
                )
                
                # Draw additional info below
                info_lines = [
                    f"Pos: ({int(seg.x)}, {int(seg.y)})",
                    f"Round: {seg.roundness:.2f}",
                    f"Area: {seg.size}"
                ]
                
                y_offset = label_pos[1] + 20
                for line in info_lines:
                    cv2.putText(
                        annotated,
                        line,
                        (label_pos[0], y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        self.color_id,
                        1
                    )
                    y_offset += 15
        
        # Draw overall statistics
        if show_info:
            info_text = f"Detected: {len(segments)} markers"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Draw info panel background
            cv2.rectangle(
                annotated,
                (10, 10),
                (400, 70),
                (0, 0, 0),
                -1
            )
            cv2.rectangle(
                annotated,
                (10, 10),
                (400, 70),
                self.color_info,
                2
            )
            
            # Draw text
            cv2.putText(
                annotated,
                info_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_info,
                2
            )
            cv2.putText(
                annotated,
                timestamp,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.color_info,
                1
            )
        
        return annotated
    
    def save_result(self,
                    image: np.ndarray,
                    segments: List[Segment],
                    filename: Optional[str] = None) -> str:
        """
        Save annotated detection result to file.
        
        Args:
            image: Input image
            segments: Detected segments
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        # Draw annotations
        annotated = self.draw_segments(image, segments)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.jpg"
        
        # Save
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), annotated)
        
        return str(output_path)
    
    def create_comparison(self,
                            original: np.ndarray,
                            annotated: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison image.
        
        Args:
            original: Original image
            annotated: Annotated image
            
        Returns:
            Combined comparison image
        """
        # Ensure same size
        if original.shape != annotated.shape:
            annotated = cv2.resize(annotated, (original.shape[1], original.shape[0]))
        
        # Create side-by-side
        comparison = np.hstack([original, annotated])
        
        # Add labels
        cv2.putText(
            comparison,
            "Original",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.putText(
            comparison,
            "Detection",
            (original.shape[1] + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        return comparison
    
    def save_comparison(self,
                        original: np.ndarray,
                        segments: List[Segment],
                        filename: Optional[str] = None) -> str:
        """
        Save side-by-side comparison.
        
        Args:
            original: Original image
            segments: Detected segments
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        annotated = self.draw_segments(original, segments)
        comparison = self.create_comparison(original, annotated)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.jpg"
        
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), comparison)
        
        return str(output_path)


def export_detection_log(segments: List[Segment],
                        output_path: str,
                        image_info: Optional[dict] = None):
    """
    Export detection results to text log file.
    
    Args:
        segments: List of detected segments
        output_path: Path to output log file
        image_info: Optional image metadata
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("WhyConID Detection Log\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Image info
        if image_info:
            f.write("Image Information:\n")
            for key, value in image_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Detection summary
        f.write(f"Total Markers Detected: {len(segments)}\n")
        f.write("\n")
        
        # Detailed results
        f.write("-" * 70 + "\n")
        f.write("Detection Details:\n")
        f.write("-" * 70 + "\n\n")
        
        for i, seg in enumerate(segments):
            f.write(f"Marker #{i + 1}:\n")
            f.write(f"  ID: {seg.ID}\n")
            f.write(f"  Position: ({seg.x:.2f}, {seg.y:.2f})\n")
            f.write(f"  Bounding Box: ({seg.minx}, {seg.miny}) -> ({seg.maxx}, {seg.maxy})\n")
            f.write(f"  Size (pixels): {seg.size}\n")
            f.write(f"  Roundness: {seg.roundness:.4f}\n")
            f.write(f"  BW Ratio: {seg.bw_ratio:.4f}\n")
            f.write(f"  Eigenvalues: m0={seg.m0:.2f}, m1={seg.m1:.2f}\n")
            f.write(f"  Valid: {seg.valid}\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("End of Detection Log\n")
        f.write("=" * 70 + "\n")


def export_detection_csv(segments: List[Segment], output_path: str):
    """
    Export detection results to CSV file.
    
    Args:
        segments: List of detected segments
        output_path: Path to output CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Marker_ID', 'Center_X', 'Center_Y', 
            'BBox_MinX', 'BBox_MinY', 'BBox_MaxX', 'BBox_MaxY',
            'Size', 'Roundness', 'BW_Ratio', 
            'M0', 'M1', 'Valid'
        ])
        
        # Data rows
        for seg in segments:
            writer.writerow([
                seg.ID,
                f"{seg.x:.2f}",
                f"{seg.y:.2f}",
                seg.minx,
                seg.miny,
                seg.maxx,
                seg.maxy,
                seg.size,
                f"{seg.roundness:.4f}",
                f"{seg.bw_ratio:.4f}",
                f"{seg.m0:.2f}",
                f"{seg.m1:.2f}",
                seg.valid
            ])
