"""
:py:mod:`pninexus.io.h5cpp` is a thin wrapper around the *h5cpp* C++ library where every
namespace in the C++ library has a counterpart as a sub-package of 
:py:mod:`pninexus.io.h5cpp`.  

"""


import pninexus.h5cpp._h5cpp 
import pninexus.h5cpp.attribute
import pninexus.h5cpp.dataspace
import pninexus.h5cpp.datatype
import pninexus.h5cpp.file
import pninexus.h5cpp.filter
import pninexus.h5cpp.node
import pninexus.h5cpp.property

from pninexus.h5cpp._h5cpp import IteratorConfig
from pninexus.h5cpp._h5cpp import IterationOrder
from pninexus.h5cpp._h5cpp import IterationIndex
from pninexus.h5cpp._h5cpp import Path
