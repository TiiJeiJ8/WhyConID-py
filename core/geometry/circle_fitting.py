"""
Circle and ellipse fitting algorithms.
Implements algebraic and geometric fitting methods.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import least_squares


def fit_circle_algebraic(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a circle using algebraic least squares method.
    
    Fast but less accurate than geometric fitting. Minimizes algebraic
    distance rather than geometric distance.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        
    Returns:
        (cx, cy, r) - center coordinates and radius
        
    Reference:
        Based on algebraic circle fitting, solving linear system for
        circle equation: x² + y² + Dx + Ey + F = 0
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    
    X = points[:, 0]
    Y = points[:, 1]
    
    # Build matrix A and vector b
    A = np.column_stack([X, Y, np.ones_like(X)])
    b = -(X**2 + Y**2)
    
    # Solve least squares: A @ [D, E, F]^T = b
    solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = solution
    
    # Convert to center-radius form
    cx = -D / 2.0
    cy = -E / 2.0
    r = np.sqrt(cx*cx + cy*cy - F)
    
    return cx, cy, r


def fit_circle_nonlinear(points: np.ndarray,
                            initial_guess: Optional[Tuple[float, float, float]] = None) -> Tuple[float, float, float]:
    """
    Fit a circle using nonlinear least squares (geometric distance).
    
    More accurate than algebraic method. Minimizes sum of squared distances
    from points to circle.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        initial_guess: Optional (cx, cy, r) initial estimate
        
    Returns:
        (cx, cy, r) - center coordinates and radius
        
    Reference:
        Uses Levenberg-Marquardt optimization to minimize:
        Σ(√((xi - cx)² + (yi - cy)²) - r)²
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    
    # Get initial guess if not provided
    if initial_guess is None:
        cx_init = np.mean(points[:, 0])
        cy_init = np.mean(points[:, 1])
        r_init = np.mean(np.sqrt(
            (points[:, 0] - cx_init)**2 + 
            (points[:, 1] - cy_init)**2
        ))
        initial_guess = (cx_init, cy_init, r_init)
    
    def residuals(params):
        cx, cy, r = params
        distances = np.sqrt(
            (points[:, 0] - cx)**2 + 
            (points[:, 1] - cy)**2
        )
        return distances - r
    
    # Optimize
    result = least_squares(residuals, initial_guess, method='lm')
    
    cx, cy, r = result.x
    return cx, cy, abs(r)


def fit_ellipse(points: np.ndarray) -> Optional[Tuple[Tuple[float, float],
                                                        Tuple[float, float],
                                                        float]]:
    """
    Fit an ellipse to points.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        
    Returns:
        ((cx, cy), (major_axis, minor_axis), angle) or None if fitting fails
        
    Note:
        Wrapper around OpenCV's fitEllipse for consistency.
        Requires at least 5 points.
    """
    if len(points) < 5:
        return None
    
    try:
        import cv2
        # OpenCV expects points as array of shape (N, 1, 2)
        points_cv = points.reshape((-1, 1, 2)).astype(np.float32)
        ellipse = cv2.fitEllipse(points_cv)
        return ellipse
    except:
        return None


def calculate_circularity(contour: np.ndarray) -> float:
    """
    Calculate circularity (compactness) of a contour.
    
    Circularity = 4π × Area / Perimeter²
    Perfect circle = 1.0, lower values indicate less circular shapes.
    
    Args:
        contour: Contour points
        
    Returns:
        Circularity value (0 to 1)
    """
    import cv2
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 0.0
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(circularity, 1.0)


def point_to_circle_distance(point: Tuple[float, float],
                                center: Tuple[float, float],
                                radius: float) -> float:
    """
    Calculate distance from a point to a circle.
    
    Args:
        point: (x, y) coordinates
        center: (cx, cy) circle center
        radius: Circle radius
        
    Returns:
        Absolute distance from point to circle edge
    """
    px, py = point
    cx, cy = center
    
    dist_to_center = np.sqrt((px - cx)**2 + (py - cy)**2)
    return abs(dist_to_center - radius)
