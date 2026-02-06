"""
Necklace-style ID generation for circular markers.
Based on WhyConID's CNecklace.cs

Implements rotation-invariant ID encoding using necklace theory.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class NecklaceInfo:
    """
    Information about a necklace ID.
    Corresponds to SNecklace in C# version.
    """
    id: int = -1
    rotation: int = 0


class CNecklace:
    """
    Necklace ID generator and decoder.
    
    Generates rotation-invariant IDs from binary sequences using
    the canonical necklace representation (lexicographically minimal rotation).
    """
    
    def __init__(self, bits: int = 5):
        """
        Initialize necklace generator.
        
        Args:
            bits: Number of bits in the necklace code (default: 5)
        """
        self.length = bits
        self.id_length = 2 ** bits
        self.id_array: List[NecklaceInfo] = []
        self.unknown = NecklaceInfo(id=-1, rotation=-1)
        
        self._generate_id_array()
        
    def _generate_id_array(self):
        """
        Generate the ID lookup table.
        
        Creates a mapping from all possible bit patterns to their canonical
        necklace IDs, accounting for rotational equivalence and symmetry.
        """
        self.id_array = [NecklaceInfo() for _ in range(self.id_length)]
        
        current_id = 1  # IDs start from 1
        
        for id_val in range(self.id_length):
            # Check if there is a lower number that could be created by bitshifting
            temp_id = id_val
            rotations = 0
            cached = [0] * self.length
            is_symmetrical = False
            
            while rotations < self.length and not is_symmetrical:
                # Rotate: shift right and move LSB to MSB
                bit = temp_id % 2
                temp_id = temp_id // 2 + bit * (2 ** (self.length - 1))
                
                if bit != 0 or id_val == 0:
                    # Check for symmetry
                    for i in range(rotations):
                        if cached[i] == temp_id:
                            is_symmetrical = True
                            break
                
                cached[rotations] = temp_id
                rotations += 1
                
                if id_val > temp_id:
                    break
            
            # Assign ID based on equivalence class
            if is_symmetrical:
                self.id_array[id_val].id = -1
                self.id_array[id_val].rotation = -1
            elif id_val > temp_id:
                if self.id_array[temp_id].id != -1:
                    self.id_array[id_val] = NecklaceInfo(
                        id=self.id_array[temp_id].id,
                        rotation=self.id_array[temp_id].rotation + rotations
                    )
                else:
                    self.id_array[id_val].rotation = rotations
            else:
                self.id_array[id_val].id = current_id
                self.id_array[id_val].rotation = 0
                current_id += 1
        
        # Special case: all ones pattern (ID 0)
        self.id_array[self.id_length - 1].id = 0
        self.id_array[self.id_length - 1].rotation = 0
        
    def get_id(self, code: int) -> NecklaceInfo:
        """
        Get necklace ID information for a given bit pattern.
        
        Args:
            code: Integer representation of the bit pattern
            
        Returns:
            NecklaceInfo with ID and rotation
        """
        if 0 <= code < self.id_length:
            return self.id_array[code]
        return self.unknown
    
    def decode_sequence(self, sequence: List[int]) -> NecklaceInfo:
        """
        Decode a binary sequence into a necklace ID.
        
        Args:
            sequence: List of binary values (0 or 1)
            
        Returns:
            NecklaceInfo with ID and rotation
        """
        if len(sequence) != self.length:
            return self.unknown
        
        # Convert sequence to integer
        code = 0
        for i, bit in enumerate(sequence):
            if bit:
                code += 2 ** i
        
        return self.get_id(code)
    
    def extract_from_points(self, points: np.ndarray, center: Tuple[float, float],
                            num_sectors: Optional[int] = None) -> NecklaceInfo:
        """
        Extract necklace ID from points around a circle.
        
        Args:
            points: Nx2 array of (x, y) coordinates
            center: (cx, cy) center of the circle
            num_sectors: Number of angular sectors (defaults to self.length)
            
        Returns:
            NecklaceInfo with decoded ID
        """
        if num_sectors is None:
            num_sectors = self.length
        
        cx, cy = center
        
        # Convert points to polar coordinates
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        angles = np.arctan2(dy, dx)
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_angles = angles[sorted_indices]
        
        # Divide into sectors
        sector_size = 2 * np.pi / num_sectors
        sequence = [0] * num_sectors
        
        for angle in sorted_angles:
            # Normalize angle to [0, 2π)
            normalized_angle = angle if angle >= 0 else angle + 2 * np.pi
            sector = int(normalized_angle / sector_size)
            if 0 <= sector < num_sectors:
                sequence[sector] = 1
        
        return self.decode_sequence(sequence)
    
    def get_canonical_rotation(self, sequence: List[int]) -> List[int]:
        """
        Get the canonical (lexicographically minimal) rotation of a sequence.
        
        Args:
            sequence: Binary sequence
            
        Returns:
            Canonical rotation of the sequence
        """
        n = len(sequence)
        min_rotation = sequence.copy()
        
        for i in range(1, n):
            rotated = sequence[i:] + sequence[:i]
            if rotated < min_rotation:
                min_rotation = rotated
        
        return min_rotation
    
    def __repr__(self):
        """String representation."""
        return f"CNecklace(bits={self.length}, unique_ids={self._count_unique_ids()})"
    
    def _count_unique_ids(self) -> int:
        """Count number of unique IDs (excluding invalid ones)."""
        unique = set()
        for info in self.id_array:
            if info.id >= 0:
                unique.add(info.id)
        return len(unique)
