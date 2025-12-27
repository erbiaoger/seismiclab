"""
Parallel Matrix Factorization module for tensor completion.

Functions
---------
pmf : Parallel matrix factorization for tensor completion
completion : Tensor completion via PMF
"""

from .pmf import pmf, completion

__all__ = [
    "pmf",
    "completion",
]
