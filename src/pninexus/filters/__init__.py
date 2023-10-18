"""
:py:mod:`pninexus.filters` is a path setter for HDF5_PLUGIN_PATH
"""
import os
from pathlib import Path

os.environ["HDF5_PLUGIN_PATH"] = os.path.split(Path(__file__))[0]
