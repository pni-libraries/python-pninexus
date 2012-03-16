#setup script for libpninx-python package
import sys
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from distutils.fancy_getopt import FancyGetopt
from distutils.fancy_getopt import fancy_getopt

#add here some options to handle additional compiler parameters
cliopts =[]
cliopts.append(("h5libdir=",None,"HDF5 library path"))
cliopts.append(("h5incdir=",None,"HDF5 include path"))
cliopts.append(("nxlibdir=",None,"PNI NX library path"))
cliopts.append(("nxincdir=",None,"PNI NX include path"))
cliopts.append(("utlibdir=",None,"PNI utilities library path"))
cliopts.append(("utincdir=",None,"PNI utilities include path"))

op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

include_dirs = []
library_dirs = []

try: include_dirs.append(opts.h5incdir)
except: pass

try: include_dirs.append(opts.nxincdir)
except: pass

try: include_dirs.append(opts.utincdir)
except: pass

try: library_dirs.append(opts.h5libdir)
except: pass

try: library_dirs.append(opts.nxlibdir)
except: pass

try: library_dirs.append(opts.utlibdir)
except: pass

#in the end we need to add the Python include directory
include_dirs.append(get_python_inc())


libs = ["boost_python","pniutils","pninx","hdf5"]
compile_args = ["-std=c++0x","-g","-O0"]
files = ["src/nx.cpp","src/NXWrapperHelpers.cpp","src/NXWrapperErrors.cpp"]

nxh5 = Extension("nxh5",files,
                 extra_compile_args = compile_args,
                 libraries=libs,
                 library_dirs=library_dirs,
                 include_dirs=include_dirs)

setup(name="libpninx-python",
        author="Eugen Wintersberger",
        author_email="eugen.wintersberger@desy.de",
        description="Python wrapper for libpninx",
        version = "0.1.0",
        ext_package="pni.nx.h5",
        ext_modules=[nxh5],
        packages = ["pni","pni.nx","pni.nx.h5"],
        url="https://sourceforge.net/projects/libpninxpython/",
        script_args = args
        )

