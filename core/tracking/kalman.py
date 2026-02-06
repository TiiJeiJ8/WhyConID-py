"""
Kalman Filter for 2D position tracking and prediction.
"""

import numpy as np
from typing import Tuple, Optional, List


class KalmanFilter2D:
    """
    2D Kalman Filter for tracking marker position and velocity.
    
    State vector: [x, y, vx, vy]
    Measurement: [x, y]
    """
    
    def __init__(self,
                    process_noise: float = 1.0,
                    measurement_noise: float = 5.0,
                    initial_uncertainty: float = 100.0):
        """
        Initialize Kalman Filter.
        
        Args:
            process_noise: Process noise covariance (how much we trust the model)
            measurement_noise: Measurement noise covariance (how much we trust measurements)
            initial_uncertainty: Initial state uncertainty
        """
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State covariance matrix
        self.P = np.eye(4) * initial_uncertainty
        
        # State transition matrix (constant velocity model)
        # x_new = x + vx * dt
        # y_new = y + vy * dt
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=float)
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        self.Q[2:, 2:] *= 0.1  # Less noise for velocity
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        self.initialized = False
        
    def initialize(self, x: float, y: float):
        """
        Initialize filter with first measurement.
        
        Args:
            x: Initial x position
            y: Initial y position
        """
        self.state = np.array([x, y, 0, 0])
        self.initialized = True
        
    def predict(self, dt: float = 1.0) -> Tuple[float, float]:
        """
        Predict next state.
        
        Args:
            dt: Time step
            
        Returns:
            Predicted (x, y) position
        """
        # Update transition matrix with time step
        F = self.F.copy()
        F[0, 2] = dt
        F[1, 3] = dt
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return self.state[0], self.state[1]
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Update filter with new measurement.
        
        Args:
            x: Measured x position
            y: Measured y position
            
        Returns:
            Corrected (x, y) position
        """
        if not self.initialized:
            self.initialize(x, y)
            return x, y
        
        # Measurement
        z = np.array([x, y])
        
        # Innovation (measurement residual)
        y_innov = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y_innov
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[0], self.state[1]
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate.
        
        Returns:
            (vx, vy) velocity
        """
        return self.state[2], self.state[3]
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current position estimate.
        
        Returns:
            (x, y) position
        """
        return self.state[0], self.state[1]
    
    def predict_future(self, n_steps: int, dt: float = 1.0) -> List[Tuple[float, float]]:
        """
        Predict future positions without updating internal state.
        
        Args:
            n_steps: Number of future steps to predict
            dt: Time step for each prediction
            
        Returns:
            List of predicted (x, y) positions
        """
        if not self.initialized:
            return []
        
        # Save current state
        saved_state = self.state.copy()
        saved_P = self.P.copy()
        
        # Predict future positions
        future_positions = []
        for _ in range(n_steps):
            # Update transition matrix
            F = self.F.copy()
            F[0, 2] = dt
            F[1, 3] = dt
            
            # Predict next state
            self.state = F @ self.state
            self.P = F @ self.P @ F.T + self.Q
            
            future_positions.append((self.state[0], self.state[1]))
        
        # Restore original state (don't affect actual tracking)
        self.state = saved_state
        self.P = saved_P
        
        return future_positions
