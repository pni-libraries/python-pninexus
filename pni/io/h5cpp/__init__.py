"""
:py:mod:`pni.io.h5cpp` is a thin wrapper around the *h5cpp* C++ library where every
namespace in the C++ library has a counterpart as a sub-package of 
:py:mod:`pni.io.h5cpp`.  

"""


import pni.io.h5cpp._h5cpp 
import pni.io.h5cpp.attribute
import pni.io.h5cpp.dataspace
import pni.io.h5cpp.datatype
import pni.io.h5cpp.file
import pni.io.h5cpp.filter
import pni.io.h5cpp.node
import pni.io.h5cpp.property

from pni.io.h5cpp._h5cpp import IteratorConfig
from pni.io.h5cpp._h5cpp import IterationOrder
from pni.io.h5cpp._h5cpp import IterationIndex
from pni.io.h5cpp._h5cpp import Path
