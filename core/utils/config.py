"""
Configuration management for WhyConID.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DetectionConfig:
    """Detection parameters configuration."""
    threshold: int = 128
    min_size: int = 20
    max_threshold: int = 255
    circular_tolerance: float = 0.3
    circularity_tolerance: float = 0.3
    ratio_tolerance: float = 0.3
    enable_corrections: bool = True


@dataclass
class CameraConfig:
    """Camera parameters configuration."""
    width: int = 640
    height: int = 480
    fps: int = 30
    focal_length: tuple = (500.0, 500.0)
    principal_point: tuple = (320.0, 240.0)
    distortion: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class MarkerConfig:
    """Marker properties configuration."""
    diameter: float = 0.05  # meters
    necklace_bits: int = 5
    num_markers: int = 1


class Config:
    """
    Main configuration manager.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to JSON config file (optional)
        """
        self.detection = DetectionConfig()
        self.camera = CameraConfig()
        self.marker = MarkerConfig()
        
        if config_path and config_path.exists():
            self.load(config_path)
    
    def load(self, path: Path):
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to config file
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        if 'detection' in data:
            self.detection = DetectionConfig(**data['detection'])
        if 'camera' in data:
            self.camera = CameraConfig(**data['camera'])
        if 'marker' in data:
            self.marker = MarkerConfig(**data['marker'])
    
    def save(self, path: Path):
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save config
        """
        data = {
            'detection': asdict(self.detection),
            'camera': asdict(self.camera),
            'marker': asdict(self.marker)
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'detection': asdict(self.detection),
            'camera': asdict(self.camera),
            'marker': asdict(self.marker)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'detection' in data:
            config.detection = DetectionConfig(**data['detection'])
        if 'camera' in data:
            config.camera = CameraConfig(**data['camera'])
        if 'marker' in data:
            config.marker = MarkerConfig(**data['marker'])
        
        return config
