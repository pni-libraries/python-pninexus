#setup script for libpninx-python package

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc

libs = ["boost_python","pniutils","pninx","hdf5"]
include_dirs = ["/home/eugen/Development/HDRI/projects/install/include",
                "/home/eugen/Development/HDRI/projects/hdf5/include",
                get_python_inc()]
library_dirs = ["/home/eugen/Development/HDRI/projects/install/lib",
                "/home/eugen/Development/HDRI/projects/hdf5/lib"]
compile_args = ["-std=c++0x","-O0","-g"]
files = ["src/nx.cpp"]


nxh5 = Extension("nxh5",files,
                 extra_compile_args = compile_args,
                 libraries=libs,
                 library_dirs=library_dirs,
                 include_dirs=include_dirs)

setup(name="PNINexus",
        author="Eugen Wintersberger",
        author_email="eugen.wintersberger@desy.de",
        description="Python wrapper for libpninx",
        version = "0.0.1",
        ext_package="pni.nx.h5",
        ext_modules=[nxh5],
        packages = ["pni","pni.nx","pni.nx.h5"]
        )

