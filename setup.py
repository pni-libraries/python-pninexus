#setup script for python-pnicore
import sys
import os
from distutils.core import setup
from distutils.extension import Extension
from distutils.fancy_getopt import FancyGetopt
from distutils.ccompiler import new_compiler
from numpy.distutils.misc_util import get_numpy_include_dirs
from pkgconfig import package
import sysconfig

#-----------------------------------------------------------------------------
# manage command line arguments
#-----------------------------------------------------------------------------
cliopts =[]
cliopts.append(("debug",None,"append debuging options"))

op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

debug = False
for o,v in op.get_option_order():
    if o == "debug":
        debug = True

#obtain the default compiler
cc = new_compiler()


#-----------------------------------------------------------------------------
# load pnicore configuration with pkg-config
#-----------------------------------------------------------------------------
pnicore        = package('pnicore')

#add the configuration to libraries, include directories and library 
#directories
include_dirs = pnicore.include_dirs
library_dirs = pnicore.library_dirs
libraries    = pnicore.libraries
libraries.append('boost_python')

#-----------------------------------------------------------------------------
# set compiler options
#-----------------------------------------------------------------------------
extra_compile_args = ['-std=c++11','-Wall','-Wextra',
                      '-fdiagnostics-show-option']
extra_compile_args.extend(pnicore.compiler_flags)
if(debug):
    extra_compile_args.append('-O0')
    extra_compile_args.append('-g')

#-----------------------------------------------------------------------------
# build shared library code
#-----------------------------------------------------------------------------
libincludes = include_dirs
libincludes.append(sysconfig.get_config_var('INCLUDEPY'))
libsources = [ "src/numpy_utils.cpp", "src/errors.cpp", "src/utils.cpp",]
libargs = extra_compile_args
libargs.append('-fPIC')
libobjects = cc.compile(libsources,
                        include_dirs=libincludes,
                        extra_preargs=extra_compile_args)
cc.link_shared_lib(libobjects,"core_api")
#-----------------------------------------------------------------------------
# list of files for the pnicore extensions
#-----------------------------------------------------------------------------
files = ["src/bool_converter.cpp",
         "src/numpy_scalar_converter.cpp",
         "src/numpy_utils.cpp",
         "src/errors.cpp",
         "src/utils.cpp",
         "src/_core.cpp"]

header_files = ["src/bool_converter.hpp",
                "src/error_utils.hpp",
                "src/numpy_scalar_converter.hpp",
                "src/numpy_utils.hpp",
                "src/utils.hpp"]

#-----------------------------------------------------------------------------
# setup for the pnicore extension
#-----------------------------------------------------------------------------
pnicore_ext = Extension("core._core",
                        files,
                        #                        define_macros=[('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
                        include_dirs = include_dirs,
                        library_dirs = library_dirs,
                        libraries = libraries,
                        language="c++",
                        extra_compile_args = extra_compile_args)

#-----------------------------------------------------------------------------
# setup for the pnicore package
#-----------------------------------------------------------------------------
setup(name="libpnicore-python",
        author="Eugen Wintersberger",
        author_email="eugen.wintersberger@desy.de",
        description="Python wrapper for libpnicore",
        long_description="This package provides some basic functionality "+
                         "which will be used by all Python extensions of "+
                         "PNI libraries",
        maintainer = "Eugen Wintersberger",
        maintainer_email = "eugen.wintersberger@desy.de",
        version = "1.0.0",
        requires = ["numpy"],
        ext_package="pni",
        ext_modules=[pnicore_ext],
        data_files=[('include/pni/core/python',header_files)],
        packages = ["pni","pni.core"],
        url="https://code.google.com/p/pni-libraries/",
        license = "GPL",
        script_args = args
        )

