"""
Main entry point for WhyConID-py core module.
Command-line interface for marker detection with trajectory tracking.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

from detectors.circle_detect import CircleDetector
from id_generation.necklace import CNecklace
from processing.image_processor import ImageProcessor
from tracking.tracker import MarkerTracker
from utils.config import Config
from utils.logger import setup_logger
from utils.visualization import (
    DetectionVisualizer,
    export_detection_log,
    export_detection_csv
)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='WhyConID Marker Detection')
    parser.add_argument('input', type=str, help='Input image or video file (or camera index)')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--debug', type=int, default=0, help='Debug level (0-3)')
    parser.add_argument('--show', action='store_true', help='Show detection results')
    parser.add_argument('--markers', type=int, default=1, help='Number of markers to track')
    parser.add_argument('--save-img', action='store_true', help='Save annotated images')
    parser.add_argument('--save-log', action='store_true', help='Save detection log')
    parser.add_argument('--save-csv', action='store_true', help='Save results as CSV')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--track', action='store_true', help='Enable trajectory tracking')
    parser.add_argument('--save-trajectory', action='store_true', help='Save trajectory data to CSV')
    parser.add_argument('--persistent-trajectory', action='store_true', help='Keep full trajectory history (no length limit)')
    parser.add_argument('--color-trajectory', action='store_true', help='Use different colors for each track')
    parser.add_argument('--show-prediction', action='store_true', help='Show trajectory prediction (arrow to next predicted position)')
    parser.add_argument('--prediction-steps', type=int, default=5, help='Number of future steps to predict (default: 5)')
    parser.add_argument('--show-prediction-error', action='store_true', help='Show prediction error (compare predicted vs actual positions)')
    
    # Tracker tuning parameters
    parser.add_argument('--match-threshold', type=float, default=350.0, help='Maximum distance for matching markers between frames (default: 350.0)')
    parser.add_argument('--max-age', type=int, default=300, help='Maximum frames to keep a track without detection (default: 300)')
    parser.add_argument('--min-hits', type=int, default=8, help='Minimum detections before track is confirmed (default: 8)')
    parser.add_argument('--memory-frames', type=int, default=500, help='Frames to remember lost tracks for ID recovery (default: 500)')
    
    # Detection preprocessing parameters
    parser.add_argument('--temporal-smooth', action='store_true', help='Enable temporal smoothing (average recent frames)')
    parser.add_argument('--smooth-frames', type=int, default=3, help='Number of frames for temporal smoothing (default: 3)')
    parser.add_argument('--smooth-weight', type=float, default=0.6, help='Weight for current frame in smoothing (0-1, default: 0.6)')
    parser.add_argument('--crop-border', type=int, default=0, help='Crop N pixels from border to remove black edges (default: 0)')
    parser.add_argument('--use-clahe', action='store_true', help='Enable CLAHE (Contrast Limited Adaptive Histogram Equalization)')
    parser.add_argument('--clahe-clip', type=float, default=2.0, help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--clahe-grid', type=int, default=8, help='CLAHE grid size (default: 8)')
    
    # Depth estimation parameters (3D positioning)
    parser.add_argument('--enable-depth', action='store_true', help='Enable depth estimation from marker size')
    parser.add_argument('--focal-length', type=float, help='Focal length in pixels (for depth estimation)')
    parser.add_argument('--fov', type=float, help='Horizontal field of view in degrees (alternative to focal length)')
    parser.add_argument('--marker-diameter', type=float, default=50.0, help='Real-world marker diameter in mm (default: 50.0)')
    
    # Camera extrinsic parameters (world coordinates)
    parser.add_argument('--camera-position', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Camera position in world frame (X Y Z in meters)')
    parser.add_argument('--camera-rotation', type=float, nargs=3, metavar=('ROLL', 'PITCH', 'YAW'),
                       help='Camera rotation in degrees (roll pitch yaw)')
    parser.add_argument('--export-3d', action='store_true', help='Export 3D trajectory visualization (HTML + PNG)')
    
    args = parser.parse_args()
    
    # Create timestamped output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    run_output_dir = base_output_dir / f"run_{timestamp}"
    
    # Setup logger
    log_file = None
    if args.save_log:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_output_dir / "detection_log.txt"
    
    logger = setup_logger('whyconid', level=10 if args.debug > 0 else 20, log_file=log_file)
    
    # Load config
    config = Config()
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config.load(config_path)
            logger.info(f"Loaded config from {config_path}")
    
    # Determine input source
    try:
        camera_index = int(args.input)
        cap = cv2.VideoCapture(camera_index)
        is_camera = True
        logger.info(f"Using camera {camera_index}")
    except ValueError:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return
        cap = cv2.VideoCapture(str(input_path))
        is_camera = False
        logger.info(f"Processing file: {input_path}")
    
    if not cap.isOpened():
        logger.error("Failed to open input source")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
    duration_sec = total_frames / fps if fps > 0 and total_frames > 0 else 0
    
    logger.info(f"Video properties: {width}x{height} @ {fps} fps")
    if total_frames > 0:
        logger.info(f"Total frames: {total_frames}, Duration: {duration_sec:.1f}s")
    
    # Initialize detector
    # Enable motion mode for video files
    motion_mode = not is_camera  # Enable for videos, disable for live camera
    detector = CircleDetector(
        width=width,
        height=height,
        num_bots=args.markers,
        debug=args.debug,
        motion_mode=motion_mode,
        use_clahe=args.use_clahe if hasattr(args, 'use_clahe') else False,
        clahe_clip=args.clahe_clip if hasattr(args, 'clahe_clip') else 2.0,
        clahe_grid=args.clahe_grid if hasattr(args, 'clahe_grid') else 8,
        temporal_smooth=args.temporal_smooth if hasattr(args, 'temporal_smooth') else False,
        smooth_frames=args.smooth_frames if hasattr(args, 'smooth_frames') else 3,
        smooth_weight=args.smooth_weight if hasattr(args, 'smooth_weight') else 0.6,
        crop_border=args.crop_border if hasattr(args, 'crop_border') else 0
    )
    
    # Log preprocessing settings
    preproc_info = []
    if hasattr(args, 'crop_border') and args.crop_border > 0:
        preproc_info.append(f"crop={args.crop_border}px")
    if hasattr(args, 'use_clahe') and args.use_clahe:
        preproc_info.append(f"CLAHE(clip={args.clahe_clip}, grid={args.clahe_grid})")
    if hasattr(args, 'temporal_smooth') and args.temporal_smooth:
        preproc_info.append(f"temporal_smooth(frames={args.smooth_frames}, weight={args.smooth_weight})")
    if motion_mode:
        preproc_info.append("motion_mode")
    
    if preproc_info:
        logger.info(f"Preprocessing enabled: {', '.join(preproc_info)}")
    else:
        logger.info("Standard detection mode (no preprocessing)")
    
    # Initialize necklace decoder
    necklace = CNecklace(bits=config.marker.necklace_bits)
    logger.info(f"Initialized necklace decoder: {necklace}")
    
    # Initialize depth estimator if enabled
    depth_estimator = None
    camera_transform = None
    
    if args.enable_depth:
        from utils.depth_estimation import DepthEstimator
        from utils.camera_transform import CameraTransform
        
        # Initialize camera transform if extrinsic parameters provided
        if args.camera_position is not None or args.camera_rotation is not None:
            cam_pos = tuple(args.camera_position) if args.camera_position else (0.0, 0.0, 1.5)
            cam_rot = tuple(args.camera_rotation) if args.camera_rotation else (0.0, 45.0, 0.0)
            camera_transform = CameraTransform(position=cam_pos, rotation=cam_rot)
            logger.info(f"Camera extrinsic params: pos={cam_pos}, rot={cam_rot}")
        
        if args.focal_length:
            depth_estimator = DepthEstimator(
                focal_length_px=args.focal_length,
                marker_diameter_mm=args.marker_diameter,
                camera_transform=camera_transform
            )
        elif args.fov:
            depth_estimator = DepthEstimator(
                fov_horizontal_deg=args.fov,
                image_width_px=width,
                marker_diameter_mm=args.marker_diameter,
                camera_transform=camera_transform
            )
        else:
            # Use default estimate
            depth_estimator = DepthEstimator(
                fov_horizontal_deg=60.0,  # Typical webcam
                image_width_px=width,
                marker_diameter_mm=args.marker_diameter,
                camera_transform=camera_transform
            )
        logger.info(f"Depth estimation enabled (marker={args.marker_diameter}mm)")
    
    # Initialize visualizer with run-specific output directory
    # Need visualizer if saving images, showing, OR outputting video
    visualizer = DetectionVisualizer(
        output_dir=str(run_output_dir),
        use_color_trajectory=args.color_trajectory
    ) if (args.save_img or args.show or args.output) else None
    if args.save_img or args.save_log or args.save_csv or args.output:
        logger.info(f"Output directory: {run_output_dir}")
    
    # Initialize tracker
    tracker = None
    if args.track or args.save_trajectory:
        # Use unlimited trajectory length if persistent mode enabled
        traj_length = None if args.persistent_trajectory else 50
        
        tracker = MarkerTracker(
            max_distance=args.match_threshold,
            max_age=args.max_age,
            min_hits=args.min_hits,
            trajectory_length=traj_length,
            memory_frames=args.memory_frames,
            depth_estimator=depth_estimator,
            image_width=width,
            image_height=height
        )
        mode_info = f"Trajectory tracking enabled (match≤{args.match_threshold}px, age≤{args.max_age}, hits≥{args.min_hits}, memory={args.memory_frames}f)"
        if args.persistent_trajectory:
            mode_info += " [persistent]"
        if args.color_trajectory:
            mode_info += " [colored]"
        logger.info(mode_info)
    
    # Output video writer (optional)
    out_writer = None
    output_video_path = None
    if args.output:
        # If output is just a filename, save to run directory
        output_path = Path(args.output)
        if output_path.parent == Path('.'):
            output_video_path = run_output_dir / output_path.name
        else:
            output_video_path = output_path
        
        # Ensure parent directory exists
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps if fps > 0 else 30.0,
            (width, height)
        )
        
        if out_writer.isOpened():
            logger.info(f"Saving annotated video to: {output_video_path}")
        else:
            logger.error(f"Failed to open video writer for: {output_video_path}")
            out_writer = None
    
    frame_count = 0
    previous_predictions = {}  # Store predictions from previous frame for error visualization
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect markers
            segments = detector.detect(frame)
            
            # Update tracker
            tracked_results = []
            trajectories = {}
            track_id_map = {}
            predictions = {}
            lost_status = {}
            
            if tracker:
                timestamp_sec = frame_count / fps if fps > 0 else frame_count / 30.0
                tracked_results = tracker.update(segments, timestamp_sec)
                trajectories = tracker.get_trajectories()
                lost_status = tracker.get_lost_status()
                
                # Get predictions if enabled
                if args.show_prediction or args.show_prediction_error:
                    predictions = tracker.get_predictions(args.prediction_steps)
                
                # Map segment index to track ID
                for track_id, seg, _ in tracked_results:
                    for i, s in enumerate(segments):
                        if s is seg:
                            track_id_map[i] = track_id
                            break
            
            # Prepare annotated frame
            display_frame = None
            if args.show or out_writer or args.save_img:
                if visualizer:
                    display_frame = visualizer.draw_segments(
                        frame,
                        segments,
                        trajectories=trajectories if tracker else None,
                        track_ids=track_id_map if tracker else None,
                        predictions=predictions if args.show_prediction else None,
                        show_prediction=args.show_prediction,
                        previous_predictions=previous_predictions if args.show_prediction_error else None,
                        show_prediction_error=args.show_prediction_error,
                        lost_status=lost_status if tracker else None,
                        tracker=tracker
                    )
                
                if args.show and display_frame is not None:
                    # Draw progress bar for video files
                    if not is_camera and total_frames > 0:
                        progress = frame_count / total_frames
                        current_time = frame_count / fps if fps > 0 else 0
                        total_time = duration_sec
                        
                        # Progress bar dimensions
                        bar_height = 6
                        bar_y = height - bar_height - 5
                        bar_width = width - 20
                        bar_x = 10
                        
                        # Draw background bar (dark gray)
                        cv2.rectangle(display_frame, (bar_x, bar_y), 
                                     (bar_x + bar_width, bar_y + bar_height), 
                                     (50, 50, 50), -1)
                        
                        # Draw progress bar (green)
                        filled_width = int(bar_width * progress)
                        cv2.rectangle(display_frame, (bar_x, bar_y), 
                                     (bar_x + filled_width, bar_y + bar_height), 
                                     (0, 200, 0), -1)
                        
                        # Draw time text
                        time_text = f"{int(current_time//60):02d}:{int(current_time%60):02d} / {int(total_time//60):02d}:{int(total_time%60):02d}  ({progress*100:.1f}%)"
                        cv2.putText(display_frame, time_text,
                                   (bar_x, bar_y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    cv2.imshow('WhyConID Detection', display_frame)
                    key = cv2.waitKey(1 if is_camera else 30)
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                
                if out_writer and display_frame is not None:
                    out_writer.write(display_frame)
            
            # Store current predictions for next frame's error visualization
            if args.show_prediction_error and predictions:
                previous_predictions = predictions.copy()
            
            # Save single frame result
            if args.save_img and not is_camera and visualizer:
                img_filename = f"frame_{frame_count:04d}_detected.jpg"
                saved_path = visualizer.save_result(frame, segments, img_filename)
                if frame_count == 1:  # Log first frame
                    logger.info(f"Saved annotated image: {saved_path}")
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames, detected {len(segments)} markers")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup resources
        cap.release()
        if out_writer:
            out_writer.release()
            if output_video_path:
                logger.info(f"Video saved successfully: {output_video_path}")
        if args.show:
            cv2.destroyAllWindows()
        
        # Export final results
        if detector.current_segments:
            # Ensure output directory exists
            run_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed log file
            if args.save_log:
                log_path = run_output_dir / "detection_results.txt"
                image_info = {
                    'Source': args.input,
                    'Resolution': f"{width}x{height}",
                    'FPS': f"{fps:.1f}" if fps > 0 else "N/A",
                    'Total Frames': frame_count,
                    'Markers Detected': len(detector.current_segments)
                }
                export_detection_log(detector.current_segments, str(log_path), image_info)
                logger.info(f"Saved detection log: {log_path}")
            
            # Save CSV
            if args.save_csv:
                csv_path = run_output_dir / "detection_results.csv"
                export_detection_csv(detector.current_segments, str(csv_path))
                logger.info(f"Saved CSV results: {csv_path}")
            
            # Save trajectory data
            if tracker and args.save_trajectory:
                trajectory_path = run_output_dir / "trajectories.csv"
                tracker.export_trajectories_csv(str(trajectory_path))
                logger.info(f"Saved trajectory data: {trajectory_path}")
                
                # Summary statistics
                full_trajs = tracker.get_full_trajectories()
                logger.info(f"Tracked {len(full_trajs)} unique markers")
                for track_id, traj in full_trajs.items():
                    logger.info(f"  Track {track_id}: {len(traj)} frames")
            
            # Export 3D visualization if enabled
            if args.export_3d and tracker and depth_estimator:
                world_trajectories = tracker.get_world_trajectories()
                if world_trajectories:
                    from utils.trajectory_3d_viz import export_trajectories_3d_html, export_trajectories_3d_matplotlib
                    
                    # HTML interactive visualization
                    html_path = run_output_dir / "trajectory_3d.html"
                    export_trajectories_3d_html(world_trajectories, str(html_path), 
                                               title="3D Trajectory - World Coordinates")
                    
                    # Static image export
                    png_path = run_output_dir / "trajectory_3d.png"
                    export_trajectories_3d_matplotlib(world_trajectories, str(png_path),
                                                     title="3D Trajectory - World Coordinates")
                    
                    logger.info(f"3D visualizations exported: {html_path.name}, {png_path.name}")
                else:
                    logger.warning("No 3D trajectories available for visualization")
            
            # Create summary file
            if args.save_log or args.save_csv or args.save_img or args.output:
                summary_path = run_output_dir / "run_summary.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"WhyConID Detection Run Summary\n")
                    f.write(f"{'=' * 50}\n\n")
                    f.write(f"Run Time: {timestamp}\n")
                    f.write(f"Input: {args.input}\n")
                    f.write(f"Resolution: {width}x{height}\n")
                    f.write(f"Total Frames: {frame_count}\n")
                    f.write(f"Markers Detected: {len(detector.current_segments)}\n\n")
                    f.write(f"Output Files:\n")
                    if args.output:
                        f.write(f"  - {output_video_path.name} (annotated video)\n")
                    if args.save_img:
                        f.write(f"  - frame_XXXX_detected.jpg (annotated images)\n")
                    if args.save_log:
                        f.write(f"  - detection_log.txt (console log)\n")
                        f.write(f"  - detection_results.txt (detailed results)\n")
                    if args.save_csv:
                        f.write(f"  - detection_results.csv (CSV export)\n")
                    if args.save_trajectory and tracker:
                        f.write(f"  - trajectories.csv (trajectory data)\n")
                    if args.export_3d and tracker and depth_estimator:
                        f.write(f"  - trajectory_3d.html (interactive 3D visualization)\n")
                        f.write(f"  - trajectory_3d.png (static 3D visualization)\n")
                        f.write(f"\nTracking Statistics:\n")
                        full_trajs = tracker.get_full_trajectories()
                        f.write(f"  Total tracks: {len(full_trajs)}\n")
                        for track_id, traj in sorted(full_trajs.items()):
                            f.write(f"  Track {track_id}: {len(traj)} frames\n")
                logger.info(f"Saved run summary: {summary_path}")
        
        logger.info(f"Processing complete. Total frames: {frame_count}")


if __name__ == '__main__':
    main()
