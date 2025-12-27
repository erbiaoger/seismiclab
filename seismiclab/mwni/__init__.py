"""
Multi-Window Noise Inversion (MWNI) module.

Functions
---------
operator_nfft : N-D Fourier operator
mwni_irls : MWNI via IRLS
mwni : Multi-window noise inversion
"""

from .mwni import operator_nfft, mwni_irls, mwni

__all__ = [
    "operator_nfft",
    "mwni_irls",
    "mwni",
]
