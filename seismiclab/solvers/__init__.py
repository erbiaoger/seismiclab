"""
Solvers for seismic inverse problems.

This module contains various optimization algorithms and solvers commonly used
in seismic data processing, including conjugate gradient methods, iterative
shrinkage-thresholding, and reweighted least squares.

Functions
---------
cgls : Conjugate gradient for least squares
cglsw : Weighted conjugate gradient for least squares
cgdot : Dot product function for CG solvers
fista : Fast iterative shrinkage-thresholding algorithm
irls : Iteratively reweighted least squares
power_method : Power iteration for eigenvalue computation
thresholding : Soft and hard thresholding operations
"""

from .cgls import cgls
from .cglsw import cglsw
from .cgdot import cgdot
from .fista import fista
from .irls import irls
from .power_method import power_method
from .thresholding import thresholding

__all__ = [
    "cgls",
    "cglsw",
    "cgdot",
    "fista",
    "irls",
    "power_method",
    "thresholding",
]
