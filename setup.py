#setup script for libpninx-python package
import sys
import os
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from distutils.fancy_getopt import FancyGetopt
from distutils.fancy_getopt import fancy_getopt
from distutils.ccompiler import new_compiler
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils import misc_util

import commands

cliopts =[]
cliopts.append(("h5libdir=",None,"HDF5 library path"))
cliopts.append(("h5incdir=",None,"HDF5 include path"))
cliopts.append(("h5libname=",None,"HDF5 library name"))
cliopts.append(("nxlibdir=",None,"PNI NX library path"))
cliopts.append(("nxincdir=",None,"PNI NX include path"))
cliopts.append(("utlibdir=",None,"PNI utilities library path"))
cliopts.append(("utincdir=",None,"PNI utilities include path"))
cliopts.append(("numpyincdir=",None,"Numpy include path"))
cliopts.append(("noforeach",None,"Set noforeach option for C++"))
cliopts.append(("debug",None,"append debuging options"))

op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

debug = False
for o,v in op.get_option_order():
    if o == "debug":
        debug = True


def pkgconfig(debug=False,*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l':
                'libraries','-D':'extra_compile_args'}
    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        if(token[:2]=="-D"):
            kw.setdefault(flag_map.get(token[:2]),[]).append(token)
        else:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])

    kw["libraries"].append("boost_python")
    kw["include_dirs"].append(misc_util.get_numpy_include_dirs()[0])
    try:
        kw["extra_compile_args"].append('-std=c++0x')
    except:
        kw["extra_compile_args"] = ["-std=c++0x"]

    if debug:
        kw["extra_compile_args"].append('-O0')
        kw["extra_compile_args"].append('-g')
    return kw


files = ["src/nx.cpp","src/NXWrapperHelpers.cpp","src/NXWrapperErrors.cpp"]

nxh5 = Extension("nxh5",files,
                 **pkgconfig(debug,'pniio'))

setup(name="libpninx-python",
        author="Eugen Wintersberger",
        author_email="eugen.wintersberger@desy.de",
        description="Python wrapper for libpninx",
        version = "0.1.2",
        ext_package="pni.io.nx.h5",
        ext_modules=[nxh5],
        packages = ["pni","pni.io","pni.io.nx","pni.io.nx.h5"],
        url="https://sourceforge.net/projects/libpninxpython/",
        script_args = args
        )

