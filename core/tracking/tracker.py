"""
Marker tracking for video sequences with trajectory prediction.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from detectors.circle_detect import Segment
from .kalman import KalmanFilter2D


@dataclass
class TrackedMarker:
    """A tracked marker with history and prediction."""
    id: int
    kalman: KalmanFilter2D
    trajectory: deque  # Recent positions
    age: int = 0  # Frames since last detection
    hits: int = 0  # Total successful detections
    misses: int = 0  # Consecutive missed detections
    last_position: Optional[Tuple[float, float]] = None
    predicted_position: Optional[Tuple[float, float]] = None
    marker_id: Optional[int] = None  # Necklace ID from detection (if available)


@dataclass
class LostTrack:
    """Memory of a recently lost track for ID recovery."""
    track_id: int
    last_position: Tuple[float, float]
    marker_id: Optional[int]
    frames_since_lost: int = 0
    hits: int = 0  # Remember hits count for faster reactivation
    last_velocity: Tuple[float, float] = (0.0, 0.0)  # Last known velocity (vx, vy)
    predicted_position: Optional[Tuple[float, float]] = None  # Kalman prediction at loss time
    

class MarkerTracker:
    """
    Multi-marker tracker with Kalman filtering and trajectory prediction.
    """
    
    def __init__(self,
                    max_distance: float = 100.0,
                    max_age: int = 60,
                    min_hits: int = 5,
                    trajectory_length: int = 50,
                    memory_frames: int = 300):
        """
        Initialize marker tracker.
        
        Args:
            max_distance: Maximum distance for matching markers between frames (increased for motion)
            max_age: Maximum frames to keep a track without detection (increased tolerance)
            min_hits: Minimum detections before track is confirmed (increased to reduce noise)
            trajectory_length: Number of positions to keep in trajectory (None for unlimited)
            memory_frames: Frames to remember lost tracks for ID recovery
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.trajectory_length = trajectory_length
        self.memory_frames = memory_frames
        self.persistent_mode = (trajectory_length is None)
        
        self.tracks: Dict[int, TrackedMarker] = {}
        self.lost_tracks: List[LostTrack] = []  # Memory of recently lost tracks
        self.next_id = 0
        self.frame_count = 0
        
        # Full trajectory history (for export)
        self.full_trajectories: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        # Format: {marker_id: [(frame, timestamp, x, y, pred_x, pred_y), ...]}
        
    def update(self, segments: List[Segment], timestamp: float = None) -> List[Tuple[int, Segment, Tuple[float, float]]]:
        """
        Update tracker with new detections.
        
        Args:
            segments: List of detected segments
            timestamp: Frame timestamp (optional)
            
        Returns:
            List of (track_id, segment, predicted_position) for confirmed tracks
        """
        self.frame_count += 1
        if timestamp is None:
            timestamp = self.frame_count / 30.0  # Assume 30 FPS
        
        # Predict all existing tracks
        for track in self.tracks.values():
            pred_x, pred_y = track.kalman.predict()
            track.predicted_position = (pred_x, pred_y)
        
        # Match detections to existing tracks
        matched_tracks, unmatched_detections = self._match_detections(segments)
        
        # Update matched tracks
        results = []
        for track_id, segment in matched_tracks:
            track = self.tracks[track_id]
            
            # Update Kalman filter
            x, y = segment.x, segment.y
            corrected_x, corrected_y = track.kalman.update(x, y)
            
            # Update track info
            track.last_position = (corrected_x, corrected_y)
            track.trajectory.append((corrected_x, corrected_y))
            track.age = 0
            track.hits += 1
            track.misses = 0
            
            # Store in full trajectory history
            if track_id not in self.full_trajectories:
                self.full_trajectories[track_id] = []
            
            pred_x, pred_y = track.predicted_position if track.predicted_position else (corrected_x, corrected_y)
            self.full_trajectories[track_id].append(
                (self.frame_count, timestamp, corrected_x, corrected_y, pred_x, pred_y)
            )
            
            # Add to results (include all tracks, mark if provisional)
            results.append((track_id, segment, track.predicted_position))
        
        # Create new tracks for unmatched detections
        # First, try to recover lost track IDs
        for segment in unmatched_detections:
            recovered_id = self._try_recover_track(segment)
            if recovered_id is not None:
                # Reactivate lost track with original ID
                self._reactivate_track(recovered_id, segment)
            else:
                # Create completely new track
                self._create_track(segment)
        
        # Update unmatched tracks (increase age, predict position)
        for track_id, track in list(self.tracks.items()):
            if track_id not in [t[0] for t in matched_tracks]:
                track.age += 1
                track.misses += 1
                
                # Remove old tracks and save to memory
                if track.age > self.max_age:
                    if track.hits >= self.min_hits:  # Only remember confirmed tracks
                        self._save_lost_track(track_id, track)
                    del self.tracks[track_id]
        
        # Clean up old lost track memories
        self.lost_tracks = [lt for lt in self.lost_tracks
                            if lt.frames_since_lost < self.memory_frames]
        for lt in self.lost_tracks:
            lt.frames_since_lost += 1
        
        return results
    
    def _match_detections(self, segments: List[Segment]) -> Tuple[List[Tuple[int, Segment]], List[Segment]]:
        """
        Match detections to existing tracks using distance.
        
        Args:
            segments: New detections
            
        Returns:
            (matched_pairs, unmatched_detections)
        """
        if not self.tracks or not segments:
            return [], segments
        
        # Build cost matrix (distance between predictions and detections)
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(segments)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            pred_x, pred_y = track.predicted_position if track.predicted_position else track.last_position
            
            if pred_x is None:
                cost_matrix[i, :] = float('inf')
                continue
                
            for j, seg in enumerate(segments):
                dist = np.sqrt((pred_x - seg.x)**2 + (pred_y - seg.y)**2)
                cost_matrix[i, j] = dist if dist < self.max_distance else float('inf')
        
        # Simple greedy matching (can be improved with Hungarian algorithm)
        matched_tracks = []
        matched_det_indices = set()
        
        while True:
            # Find minimum cost
            min_val = cost_matrix.min()
            if min_val == float('inf'):
                break
                
            i, j = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
            
            matched_tracks.append((track_ids[i], segments[j]))
            matched_det_indices.add(j)
            
            # Invalidate this row and column
            cost_matrix[i, :] = float('inf')
            cost_matrix[:, j] = float('inf')
        
        # Unmatched detections
        unmatched_detections = [seg for i, seg in enumerate(segments) if i not in matched_det_indices]
        
        return matched_tracks, unmatched_detections
    
    def _create_track(self, segment: Segment):
        """Create new track for detection."""
        kalman = KalmanFilter2D()
        kalman.initialize(segment.x, segment.y)
        
        # Use deque with maxlen for limited trajectory, or list for unlimited
        if self.persistent_mode:
            trajectory = []  # Unlimited list
        else:
            trajectory = deque(maxlen=self.trajectory_length)
        
        track = TrackedMarker(
            id=self.next_id,
            kalman=kalman,
            trajectory=trajectory,
            last_position=(segment.x, segment.y),
            hits=1,
            marker_id=segment.ID if hasattr(segment, 'ID') and segment.ID >= 0 else None
        )
        track.trajectory.append((segment.x, segment.y))
        
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _try_recover_track(self, segment: Segment) -> Optional[int]:
        """
        Try to match new detection to a recently lost track using multi-feature matching.
        
        Matching features:
        1. Position distance (基础距离)
        2. Velocity direction consistency (速度方向一致性)
        3. Kalman prediction confidence (预测置信度)
        4. Marker ID if available (外观ID)
        
        Args:
            segment: New detection
            
        Returns:
            Track ID if match found, None otherwise
        """
        if not self.lost_tracks:
            return None
        
        best_match_id = None
        best_score = 0.0  # Higher score = better match
        
        for lost in self.lost_tracks:
            # Skip if lost too long ago
            if lost.frames_since_lost > self.memory_frames // 2:  # Half of memory window
                continue
            
            # Feature 1: Position distance
            # Use predicted position if available, otherwise last known position
            reference_pos = lost.predicted_position if lost.predicted_position else lost.last_position
            
            # Extrapolate position based on velocity and time lost
            extrapolated_x = lost.last_position[0] + lost.last_velocity[0] * lost.frames_since_lost
            extrapolated_y = lost.last_position[1] + lost.last_velocity[1] * lost.frames_since_lost
            
            # Distance to extrapolated position
            dist_extrap = np.sqrt((segment.x - extrapolated_x)**2 + (segment.y - extrapolated_y)**2)
            # Distance to last known position
            dist_last = np.sqrt((segment.x - lost.last_position[0])**2 + 
                               (segment.y - lost.last_position[1])**2)
            
            # Use minimum of both distances (more forgiving)
            dist = min(dist_extrap, dist_last)
            
            # Reject if too far even for recovery
            max_recovery_distance = self.max_distance * 3.0  # Allow 3x normal distance
            if dist > max_recovery_distance:
                continue
            
            # Distance score (closer = higher score, normalized to 0-1)
            # Score decays exponentially with distance
            distance_score = np.exp(-dist / self.max_distance)
            
            # Feature 2: Velocity direction consistency
            velocity_score = 0.0
            speed = np.sqrt(lost.last_velocity[0]**2 + lost.last_velocity[1]**2)
            if speed > 1.0:  # Only if object was moving
                # Calculate direction from last position to new detection
                dx = segment.x - lost.last_position[0]
                dy = segment.y - lost.last_position[1]
                new_speed = np.sqrt(dx**2 + dy**2) / max(lost.frames_since_lost, 1)
                
                if new_speed > 0.1:
                    # Normalize velocity vectors
                    vx_norm = lost.last_velocity[0] / speed
                    vy_norm = lost.last_velocity[1] / speed
                    dx_norm = dx / (new_speed * max(lost.frames_since_lost, 1))
                    dy_norm = dy / (new_speed * max(lost.frames_since_lost, 1))
                    
                    # Dot product (cosine similarity) of direction vectors
                    direction_similarity = vx_norm * dx_norm + vy_norm * dy_norm
                    # Map from [-1, 1] to [0, 1], with 1 = same direction
                    velocity_score = (direction_similarity + 1.0) / 2.0
            
            # Feature 3: Kalman prediction confidence
            # Score based on how recently the track was lost (fresher = more confident)
            freshness_score = 1.0 - (lost.frames_since_lost / (self.memory_frames // 2))
            freshness_score = max(0.0, freshness_score)  # Clamp to [0, 1]
            
            # Feature 4: Marker ID match (if available)
            id_match_bonus = 0.0
            if hasattr(segment, 'ID') and segment.ID >= 0 and lost.marker_id is not None:
                if segment.ID == lost.marker_id:
                    id_match_bonus = 1.0  # Strong bonus for ID match
            
            # Combined score (weighted sum)
            # Weights: distance=0.4, velocity=0.2, freshness=0.2, id=0.2
            score = (0.4 * distance_score + 
                    0.2 * velocity_score + 
                    0.2 * freshness_score + 
                    0.2 * id_match_bonus)
            
            # Boost score if ID matches
            if id_match_bonus > 0:
                score *= 1.5  # 50% boost for ID match
            
            # Update best match
            if score > best_score:
                best_match_id = lost.track_id
                best_score = score
        
        # Only accept match if score is above threshold
        score_threshold = 0.3  # Minimum confidence required
        if best_match_id is not None and best_score >= score_threshold:
            # Remove from lost tracks
            self.lost_tracks = [lt for lt in self.lost_tracks if lt.track_id != best_match_id]
            return best_match_id
        
        return None
    
    def _reactivate_track(self, track_id: int, segment: Segment):
        """Reactivate a lost track with new detection."""
        kalman = KalmanFilter2D()
        kalman.initialize(segment.x, segment.y)
        
        # Find the lost track to get its previous hits count
        old_hits = self.min_hits  # Default to min_hits for instant confirmation
        for lost in self.lost_tracks:
            if lost.track_id == track_id:
                old_hits = max(lost.hits, self.min_hits)  # Ensure it's immediately confirmed
                break
        
        # Use deque with maxlen for limited trajectory, or list for unlimited
        if self.persistent_mode:
            trajectory = []  # Unlimited list
        else:
            trajectory = deque(maxlen=self.trajectory_length)
        
        track = TrackedMarker(
            id=track_id,  # Use original ID!
            kalman=kalman,
            trajectory=trajectory,
            last_position=(segment.x, segment.y),
            hits=old_hits,  # Restore previous hits count (or min_hits)
            marker_id=segment.ID if hasattr(segment, 'ID') and segment.ID >= 0 else None
        )
        track.trajectory.append((segment.x, segment.y))
        
        self.tracks[track_id] = track
    
    def _save_lost_track(self, track_id: int, track: TrackedMarker):
        """Save track to memory when it's lost."""
        if track.last_position:
            # Extract velocity from Kalman state if available
            vx, vy = 0.0, 0.0
            if track.kalman.initialized and len(track.kalman.state) >= 4:
                vx = track.kalman.state[2]
                vy = track.kalman.state[3]
            
            lost = LostTrack(
                track_id=track_id,
                last_position=track.last_position,
                marker_id=track.marker_id,
                frames_since_lost=0,
                hits=track.hits,
                last_velocity=(vx, vy),
                predicted_position=track.predicted_position
            )
            self.lost_tracks.append(lost)
    
    def get_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Get recent trajectories for confirmed tracks only.
        
        Returns:
            Dict mapping track_id to list of (x, y) positions
        """
        trajectories = {}
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits:  # Only show confirmed tracks
                trajectories[track_id] = list(track.trajectory)
        return trajectories
    
    def get_predictions(self, n_steps: int = 1) -> Dict[int, List[Tuple[float, float]]]:
        """
        Get future position predictions for all confirmed tracks.
        
        Args:
            n_steps: Number of future steps to predict
            
        Returns:
            Dict mapping track_id to list of predicted (x, y) positions
        """
        predictions = {}
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits and track.kalman.initialized:
                future_positions = track.kalman.predict_future(n_steps)
                if future_positions:
                    predictions[track_id] = future_positions
        return predictions
    
    def get_lost_status(self) -> Dict[int, bool]:
        """
        Get lost/tracking status for all confirmed tracks.
        
        Returns:
            Dict mapping track_id to is_lost (True if lost/predicting, False if actively detected)
        """
        lost_status = {}
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits:  # Only for confirmed tracks
                lost_status[track_id] = (track.age > 0)  # Lost if age > 0
        return lost_status
    
    def get_full_trajectories(self) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
        """
        Get complete trajectory history for export.
        
        Returns:
            Dict mapping track_id to list of (frame, timestamp, x, y, pred_x, pred_y)
        """
        return self.full_trajectories
    
    def export_trajectories_csv(self, filepath: str):
        """
        Export full trajectories to CSV file.
        
        Args:
            filepath: Output CSV file path
        """
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Track_ID', 'Frame', 'Timestamp',
                'X', 'Y',
                'Predicted_X', 'Predicted_Y',
                'Velocity_X', 'Velocity_Y'
            ])
            
            # Data
            for track_id, trajectory in sorted(self.full_trajectories.items()):
                for i, (frame, timestamp, x, y, pred_x, pred_y) in enumerate(trajectory):
                    # Calculate velocity
                    if i > 0:
                        prev_frame, prev_ts, prev_x, prev_y, _, _ = trajectory[i-1]
                        dt = timestamp - prev_ts
                        vx = (x - prev_x) / dt if dt > 0 else 0
                        vy = (y - prev_y) / dt if dt > 0 else 0
                    else:
                        vx, vy = 0, 0
                    
                    writer.writerow([
                        track_id, frame, f"{timestamp:.3f}",
                        f"{x:.2f}", f"{y:.2f}",
                        f"{pred_x:.2f}", f"{pred_y:.2f}",
                        f"{vx:.2f}", f"{vy:.2f}"
                    ])
