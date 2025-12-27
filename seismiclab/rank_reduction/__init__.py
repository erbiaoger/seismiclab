"""
Rank reduction methods for seismic data processing.

This module contains algorithms for low-rank matrix approximations commonly
used in seismic data denoising and interpolation.

Functions
---------
cur : CUR decomposition for low-rank approximation
cur_Old : CUR decomposition (old version)
rand_svd : Randomized SVD for low-rank approximation
rqrd : Randomized QR decomposition for low-rank approximation
"""

from .cur import cur
from .cur_Old import cur_Old
from .rand_svd import rand_svd
from .rqrd import rqrd

__all__ = [
    "cur",
    "cur_Old",
    "rand_svd",
    "rqrd",
]
