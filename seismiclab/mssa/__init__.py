"""
Multichannel Singular Spectrum Analysis (MSSA) for seismic data processing.

This module contains MSSA algorithms for seismic data denoising and reconstruction.
MSSA is effective at removing random noise while preserving seismic signals.

Functions
---------
mssa_2d : 2D MSSA filtering
mssa_3d : 3D MSSA filtering
mssa_3d_interp : 3D MSSA with interpolation
"""

from .mssa_2d import mssa_2d
from .mssa_3d import mssa_3d
from .mssa_3d_interp import mssa_3d_interp

__all__ = [
    "mssa_2d",
    "mssa_3d",
    "mssa_3d_interp",
]
