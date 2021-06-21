"""
:py:mod:`pninexus.io.h5cpp` is a thin wrapper around the *h5cpp* C++
library where every
namespace in the C++ library has a counterpart as a sub-package of
:py:mod:`pninexus.io.h5cpp`
"""


from pninexus.h5cpp import _h5cpp
from pninexus.h5cpp import attribute
from pninexus.h5cpp import dataspace
from pninexus.h5cpp import datatype
from pninexus.h5cpp import file
from pninexus.h5cpp import filter
from pninexus.h5cpp import node
from pninexus.h5cpp import property

from pninexus.h5cpp._h5cpp import IteratorConfig
from pninexus.h5cpp._h5cpp import IterationOrder
from pninexus.h5cpp._h5cpp import IterationIndex
from pninexus.h5cpp._h5cpp import Path
from pninexus.h5cpp._h5cpp import print_hdf5_errors
from pninexus.h5cpp._h5cpp import current_library_version

__all__ = ["IteratorConfig", "IterationOrder", "IterationIndex", "Path",
           "print_hdf5_errors", "current_library_version",
           "attribute", "dataspace", "datatype", "file", "filter", "node",
           "property", "_h5cpp"]
