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
    
    logger.info(f"Video properties: {width}x{height} @ {fps} fps")
    
    # Initialize detector
    # Enable motion mode for video files
    motion_mode = not is_camera  # Enable for videos, disable for live camera
    detector = CircleDetector(
        width=width,
        height=height,
        num_bots=args.markers,
        debug=args.debug,
        motion_mode=motion_mode
    )
    if motion_mode:
        logger.info("Motion-aware detection enabled for video processing")
    
    # Initialize necklace decoder
    necklace = CNecklace(bits=config.marker.necklace_bits)
    logger.info(f"Initialized necklace decoder: {necklace}")
    
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
            max_distance=150.0,  # Increased for fast-moving markers
            max_age=90,          # Keep tracks longer to avoid fragmentation
            min_hits=8,          # More hits required to confirm track (reduce noise)
            trajectory_length=traj_length,
            memory_frames=300    # Remember lost tracks for 10 seconds at 30fps
        )
        mode_info = "Trajectory tracking enabled"
        if args.persistent_trajectory:
            mode_info += " (persistent mode)"
        if args.color_trajectory:
            mode_info += " (colored trajectories)"
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
            
            if tracker:
                timestamp_sec = frame_count / fps if fps > 0 else frame_count / 30.0
                tracked_results = tracker.update(segments, timestamp_sec)
                trajectories = tracker.get_trajectories()
                
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
                        show_prediction_error=args.show_prediction_error
                    )
                
                if args.show and display_frame is not None:
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
                        f.write(f"\nTracking Statistics:\n")
                        full_trajs = tracker.get_full_trajectories()
                        f.write(f"  Total tracks: {len(full_trajs)}\n")
                        for track_id, traj in sorted(full_trajs.items()):
                            f.write(f"  Track {track_id}: {len(traj)} frames\n")
                logger.info(f"Saved run summary: {summary_path}")
        
        logger.info(f"Processing complete. Total frames: {frame_count}")


if __name__ == '__main__':
    main()
