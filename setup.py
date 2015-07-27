#setup script for python-pnicore
from __future__ import print_function
import sys
import os
from numpy.distutils.misc_util import get_numpy_include_dirs
from pkgconfig import package
from setuptools import setup, find_packages, Extension
import sysconfig

def get_build_dir():
    build_dir = "lib.{platform}-{version[0]}.{version[1]}"

    return os.path.join("build",build_dir.format(platform = sysconfig.get_platform(),
                                                 version = sys.version_info))


#-----------------------------------------------------------------------------
# load pnicore configuration with pkg-config
#-----------------------------------------------------------------------------
pnicore        = package('pnicore')

#add the configuration to libraries, include directories and library 
#directories
include_dirs = pnicore.include_dirs
library_dirs = pnicore.library_dirs
libraries    = pnicore.libraries
libraries.append('boost_python-py{version[0]}{version[1]}'.format(version=sys.version_info))

#-----------------------------------------------------------------------------
# set compiler options
#-----------------------------------------------------------------------------
extra_compile_args = ['-std=c++11','-Wall','-Wextra',
                      '-fdiagnostics-show-option']
extra_compile_args.extend(pnicore.compiler_flags)

#-----------------------------------------------------------------------------
# list of files for the pnicore extensions
#-----------------------------------------------------------------------------
files = ["src/bool_converter.cpp",
         "src/numpy_scalar_converter.cpp",
         "src/errors.cpp",
         "src/numpy_utils.cpp",
         "src/utils.cpp",
         "src/init_numpy.cpp",
         "src/_core.cpp"]

header_files = ["src/bool_converter.hpp",
                "src/error_utils.hpp",
                "src/numpy_scalar_converter.hpp",
                "src/numpy_utils.hpp",
                "src/init_numpy.hpp",
                "src/utils.hpp"]

#-----------------------------------------------------------------------------
# setup for the pnicore extension
#-----------------------------------------------------------------------------
pnicore_ext = Extension("pni.core._core",
                        files,
                        include_dirs = include_dirs,
                        library_dirs = library_dirs,
                        libraries = libraries,
                        language="c++",
                        extra_compile_args = extra_compile_args)

#need to build some extra test modules
ex_trans_test = Extension("test.ex_trans_test",
                          ["test/ex_trans_test.cpp"],
                          language="c++",
                          include_dirs = include_dirs,
                          library_dirs = library_dirs,
                          libraries = libraries,
                          extra_compile_args = extra_compile_args)

object_dir = os.path.join(get_build_dir(),"pni","core")
core_name = ":_core"
soabi = sysconfig.get_config_var("SOABI")
if soabi: 
    core_name += "."+soabi+".so"
else:
    core_name += ".so"

utils_test = Extension("test.utils_test",
                          ["test/utils_test.cpp"],
                          language="c++",
                          include_dirs = include_dirs,
                          library_dirs = library_dirs+[object_dir],
                          libraries = libraries+[core_name],
                          extra_compile_args = extra_compile_args)

numpy_utils_test = Extension("test.numpy_utils_test",
                             ["test/numpy_utils_test.cpp",
                              "test/check_type_id_from_object.cpp",
                              "test/check_type_str_from_object.cpp"
                             ],
                             language="c++",
                             include_dirs = include_dirs,
                             library_dirs = library_dirs+[object_dir],
                             libraries = libraries+[core_name],
                             extra_compile_args = extra_compile_args)
#-----------------------------------------------------------------------------
# setup for the pnicore package
#-----------------------------------------------------------------------------
setup(name="python-pnicore",
      author="Eugen Wintersberger",
      author_email="eugen.wintersberger@desy.de",
      description="Python wrapper for libpnicore",
      long_description="This package provides some basic functionality "+
                       "which will be used by all Python extensions of "+
                       "PNI libraries",
      maintainer = "Eugen Wintersberger",
      maintainer_email = "eugen.wintersberger@desy.de",
      license = "GPLv2",
      version = "1.0.0",
        requires = ["numpy"],
        ext_modules=[pnicore_ext,ex_trans_test,utils_test,numpy_utils_test],
      data_files=[('include/pni/core/python',header_files)],
      packages = find_packages(),
      url="https://github.com/pni-libraries/python-pnicore",
      test_suite="test",
      test_loader = "unittest:TestLoader"
        )

