"""
WhyConID-py Core Module
A Python implementation of WhyConID circular marker detection and identification system.
"""

__version__ = "0.1.0"
__author__ = "WhyConID-py Team"

from .detectors.circle_detect import CircleDetector
from .id_generation.necklace import CNecklace
from .geometry.circle_fitting import fit_circle_algebraic, fit_circle_nonlinear

__all__ = [
    'CircleDetector',
    'CNecklace',
    'fit_circle_algebraic',
    'fit_circle_nonlinear',
]
