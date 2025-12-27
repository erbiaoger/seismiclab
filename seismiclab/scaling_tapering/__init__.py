"""
Scaling and tapering functions for seismic data.

Functions
---------
taper : Apply triangular taper to traces
gain : Apply gain to seismic traces
envelope : Compute envelope via Hilbert transform
"""

from .taper import taper, gain, envelope

__all__ = [
    "taper",
    "gain",
    "envelope",
]
