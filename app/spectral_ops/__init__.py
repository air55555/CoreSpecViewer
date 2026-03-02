"""
CoreSpecViewer spectral operations package.

This package contains the core hyperspectral “maths” used throughout the
application (IO helpers, preprocessing, continuum removal, feature extraction,
clustering, and correlation utilities).

Design notes
------------
- Functions here are intended to be importable from multiple layers (models,
  interface, tests). Keep them side-effect free where possible.

- This file is intentionally minimal: `spectral_ops` is treated as a namespace
  package, and modules are imported explicitly at call sites to keep import time
  low and dependencies obvious.
"""