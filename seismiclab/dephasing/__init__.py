"""
Dephasing module for seismic data processing.

Functions
---------
phase_correction : Apply constant phase correction
kurtosis_of_traces : Compute kurtosis of traces
"""

from .phase_correction import phase_correction, kurtosis_of_traces

__all__ = [
    "phase_correction",
    "kurtosis_of_traces",
]
