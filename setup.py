#setup script for python-pnicore
from __future__ import print_function
import sys
import os
from numpy.distutils.misc_util import get_numpy_include_dirs
from pkgconfig import package
from setuptools import setup, find_packages, Extension
from distutils.command.install import install
import sysconfig

def get_build_dir():
    build_dir = "lib.{platform}-{version[0]}.{version[1]}"

    return os.path.join("build",build_dir.format(platform = sysconfig.get_platform(),
                                                 version = sys.version_info))

#-----------------------------------------------------------------------------
# load pnicore configuration with pkg-config
#-----------------------------------------------------------------------------
pnicore  = package('pnicore')
pniio    = package('pniio')

#add the configuration to libraries, include directories and library 
#directories
include_dirs = pnicore.include_dirs + pniio.include_dirs
library_dirs = pnicore.library_dirs + pniio.library_dirs
libraries    = pnicore.libraries + pniio.libraries
libraries.append('boost_python-py{version[0]}{version[1]}'.format(version=sys.version_info))

#-----------------------------------------------------------------------------
# set compiler options
#-----------------------------------------------------------------------------
extra_compile_args = ['-std=c++11','-Wall','-Wextra',
                      '-fdiagnostics-show-option',
                      '-Wno-strict-prototypes']
extra_compile_args.extend(pnicore.compiler_flags)

#-----------------------------------------------------------------------------
# list of files for the pnicore extensions
#-----------------------------------------------------------------------------
core_files = ["src/core/bool_converter.cpp",
              "src/core/numpy_scalar_converter.cpp",
              "src/core/errors.cpp",
              "src/core/numpy_utils.cpp",
              "src/core/utils.cpp",
              "src/core/init_numpy.cpp",
              "src/core/_core.cpp"]

nxh5_files = ["src/io/_nxh5.cpp",
              "src/io/utils.cpp",
              "src/core/numpy_utils.cpp",
              "src/core/init_numpy.cpp",
              "src/core/utils.cpp"
             ]

io_files = ["src/io/_io.cpp","src/io/errors.cpp"]

core_lib_dir=os.path.join(get_build_dir(),"pni","core")

#-----------------------------------------------------------------------------
# setup for the core extension
#-----------------------------------------------------------------------------
core_ext = Extension("pni.core._core",
                     core_files,
                     include_dirs = include_dirs,
                     library_dirs = library_dirs,
                     libraries = libraries,
                     language="c++",
                     extra_compile_args = extra_compile_args)

nxh5_ext = Extension("pni.io.nx.h5._nxh5",nxh5_files,
                     include_dirs = include_dirs+["src/"],
                     library_dirs = library_dirs,
                     libraries = libraries,
                     language="c++",
                     extra_compile_args = extra_compile_args)

io_ext = Extension("pni.io._io",io_files,
                   include_dirs = include_dirs+["src/"],
                   library_dirs = library_dirs,
                   libraries = libraries,
                   #runtime_library_dirs=[core_path],
                   language="c++",
                   extra_compile_args = extra_compile_args)

#------------------------------------------------------------------------------
# setup for core test extensions
#------------------------------------------------------------------------------
#need to build some extra test modules
ex_trans_test = Extension("test.core.ex_trans_test",
                          ["test/core/ex_trans_test.cpp"],
                          language="c++",
                          include_dirs = include_dirs,
                          library_dirs = library_dirs,
                          libraries = libraries,
                          extra_compile_args = extra_compile_args)


utils_test = Extension("test.core.utils_test",
                      ["test/core/utils_test.cpp"],
                       language="c++",
                       include_dirs = include_dirs+["src/"],
                       library_dirs = library_dirs+[core_lib_dir],
                       libraries = libraries+[":_core.so"],
                       extra_compile_args = extra_compile_args)

numpy_utils_test = Extension("test.core.numpy_utils_test",
                             ["test/core/numpy_utils_test.cpp",
                              "test/core/check_type_id_from_object.cpp",
                              "test/core/check_type_str_from_object.cpp"
                             ],
                             language="c++",
                             include_dirs = include_dirs+["src/"],
                             library_dirs = library_dirs+[core_lib_dir],
                             libraries = libraries+[":_core.so"],
                             extra_compile_args = extra_compile_args)


#-----------------------------------------------------------------------------
# customized version of the `install` command. The original version shipped 
# with distutils does not install header files even if they are mentioned
# in the project setup. 
#
# We thus call `install_headers` manually.
#-----------------------------------------------------------------------------
class pnicore_install(install):
    def run(self):
        install.run(self)


#-----------------------------------------------------------------------------
# setup for the pnicore package
#-----------------------------------------------------------------------------

setup(name="pnicore",
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
      ext_modules=[core_ext,nxh5_ext,io_ext,
                   ex_trans_test,utils_test,numpy_utils_test],
      packages = find_packages(),
      url="https://github.com/pni-libraries/python-pni",
      test_suite="test",
      test_loader = "unittest:TestLoader",
      cmdclass={"install":pnicore_install}
    )

