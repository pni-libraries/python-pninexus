#setup script for python-pni
from __future__ import print_function
import sys
import os
import os.path
from ConanConfig import ConanConfig
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
# set compiler options
#-----------------------------------------------------------------------------
if sys.platform == "win32":
    extra_compile_args = []
else:
    extra_compile_args = ['-std=c++11','-Wall','-Wextra',
                          '-fdiagnostics-show-option',
                          '-Wno-strict-prototypes']

#------------------------------------------------------------------------------
# set include directories, linker path and libraries
# we have two possibilities to configure the build: 
# -> via pkg-config using locally installed libraries
# -> via Conan 
# if a conanbuildinfo.txt file exists in the root directory of the package the
# conan variant will be used automatically.
#------------------------------------------------------------------------------
core_extra_link_args = []
nxh5_extra_link_args = []
io_extra_link_args = []
nx_extra_link_args = []
nexus_extra_link_args =[]

if os.path.exists("conanbuildinfo.txt"):
    c = ConanConfig("conanbuildinfo.txt")
    include_dirs = c.includedirs
    library_dirs = c.libdirs
    libraries = c.libs
    extra_link_args = []
    for libdir in c.libdirs:
        extra_link_args.append('-Wl,-rpath,'+libdir)
    core_extra_link_args.append("-Wl,-rpath,'$ORIGIN'/../libs")
    nxh5_extra_link_args.append("-Wl,-rpath,'$ORIGIN'/../../../libs")
    nexus_extra_link_args.append("-Wl,-rpath,'$ORIGIN'/../../libs")
    io_extra_link_args.append("-Wl,-rpath,'$ORIGIN'/../libs")
    nx_extra_link_args.append("-Wl,-rpath,'$ORIGIN'/../../libs")
    
    print("linking with libraries:")
    for lib in libraries:
        print(lib)
else:
    #--------------------------------------------------------------------------
    # load pni configuration with pkg-config
    #--------------------------------------------------------------------------
    pnicore  = package('pnicore')
    pniio    = package('pniio')

    #add the configuration to libraries, include directories and library 
    #directories
    include_dirs = pnicore.include_dirs + pniio.include_dirs
    library_dirs = pnicore.library_dirs + pniio.library_dirs
    libraries    = pnicore.libraries + pniio.libraries
    libraries.append('boost_python-py{version[0]}{version[1]}'.format(version=sys.version_info))
    extra_compile_args.extend(pnicore.compiler_flags)
    extra_link_args = []

include_dirs += get_numpy_include_dirs()
#-----------------------------------------------------------------------------
# list of files for the pnicore extensions
#-----------------------------------------------------------------------------
common_sources = ["src/core/numpy_utils.cpp",
                  "src/core/utils.cpp",
                  # "src/core/init_numpy.cpp"
                  ]

core_files = ["src/core/bool_converter.cpp",
              "src/core/numpy_scalar_converter.cpp",
              "src/core/errors.cpp",
              "src/core/_core.cpp"]+common_sources

nxh5_files = [
             "src/io/_nxh5.cpp",
             "src/io/utils.cpp",
             "src/io/nxattribute_manager_wrapper.cpp",
             "src/io/nxattribute_wrapper.cpp",
             "src/io/nxfile_wrapper.cpp",
             "src/io/node_to_python.cpp",
             # "src/io/nxfield_wrapper.cpp",
             "src/io/nxgroup_wrapper.cpp",
             # "src/io/xml_functions_wrapper.cpp",
             "src/io/io_operations.cpp"
             ]+common_sources
             
nexus_files = [
            "src/io/_nexus.cpp",
            "src/io/nexus/boost_filesystem_path_conversion.cpp",
            "src/io/nexus/file_wrapper.cpp",
            "src/io/nexus/errors.cpp",
            "src/io/nexus/attribute_manager_wrapper.cpp"
            ]

io_files = ["src/io/_io.cpp","src/io/errors.cpp"]

nx_files = [
            "src/io/_nx.cpp",
            "src/io/nxpath/boost_path_to_object.cpp",
            "src/io/nxpath/element_dict_converter.cpp",
            "src/io/nxpath/nxpath_wrapper.cpp",
            ]

core_lib_dir=os.path.join(get_build_dir(),"pni","core")

#-----------------------------------------------------------------------------
# setup for the core extension
#-----------------------------------------------------------------------------
core_ext = Extension("pni.core._core",
                     core_files,
                     include_dirs = include_dirs,
                     library_dirs = library_dirs,
                     libraries = libraries,
                     extra_link_args = core_extra_link_args,
                     language="c++",
                     extra_compile_args = extra_compile_args)

nxh5_ext = Extension("pni.io.nx.h5._nxh5",nxh5_files,
                     include_dirs = include_dirs+["src/"],
                     library_dirs = library_dirs,
                     libraries = libraries,
                     extra_link_args = nxh5_extra_link_args,
                     language="c++",
                     extra_compile_args = extra_compile_args)

nexus_ext = Extension("pni.io.nexus._nexus",nexus_files,
                     include_dirs = include_dirs+["src/"],
                     library_dirs = library_dirs,
                     libraries = libraries,
                     extra_link_args = nexus_extra_link_args,
                     language="c++",
                     extra_compile_args = extra_compile_args)

io_ext = Extension("pni.io._io",io_files,
                   include_dirs = include_dirs+["src/"],
                   library_dirs = library_dirs,
                   libraries = libraries,
                   extra_link_args = io_extra_link_args,
                   language="c++",
                   extra_compile_args = extra_compile_args)

nx_ext = Extension("pni.io.nx._nx",nx_files,
                   include_dirs = include_dirs+["src/"],
                   library_dirs = library_dirs,
                   extra_link_args = nx_extra_link_args,
                   libraries = libraries,
                   language="c++",
                   extra_compile_args = extra_compile_args)

#------------------------------------------------------------------------------
# setup for core test extensions
#------------------------------------------------------------------------------
#need to build some extra test modules
ex_trans_test = Extension("pni.test.core.ex_trans_test",
                          ["pni/test/core/ex_trans_test.cpp"],
                          language="c++",
                          include_dirs = include_dirs,
                          library_dirs = library_dirs,
                          extra_link_args = extra_link_args,
                          libraries = libraries,
                          extra_compile_args = extra_compile_args)


utils_test = Extension("pni.test.core.utils_test",
                      ["pni/test/core/utils_test.cpp"]+common_sources,
                       language="c++",
                       include_dirs = include_dirs+["src/"],
                       library_dirs = library_dirs,
                       extra_link_args = extra_link_args,
                       libraries = libraries,
                       extra_compile_args = extra_compile_args)

numpy_utils_test = Extension("pni.test.core.numpy_utils_test",
                             ["pni/test/core/numpy_utils_test.cpp",
                              "pni/test/core/check_type_id_from_object.cpp",
                              "pni/test/core/check_type_str_from_object.cpp"
                             ]+common_sources,
                             language="c++",
                             include_dirs = include_dirs+["src/"], 
                             library_dirs = library_dirs,
                             extra_link_args = extra_link_args,
                             libraries = libraries,
                             extra_compile_args = extra_compile_args)


#-----------------------------------------------------------------------------
# customized version of the `install` command. The original version shipped 
# with distutils does not install header files even if they are mentioned
# in the project setup. 
#
# We thus call `install_headers` manually.
#-----------------------------------------------------------------------------
class pni_install(install):
    def run(self):
        install.run(self)


#-----------------------------------------------------------------------------
# setup for the pnicore package
#-----------------------------------------------------------------------------

setup(name="pni",
      author="Eugen Wintersberger",
      author_email="eugen.wintersberger@desy.de",
      description="Python wrapper for the PNI libraries",
      long_description="This package provides wrappers for the PNI C++ "+
                       "libraries libpnicore and libpniio.",
      maintainer = "Eugen Wintersberger",
      maintainer_email = "eugen.wintersberger@desy.de",
      license = "GPLv2",
      version = "1.1.0",
      requires = ["numpy"],
      ext_modules=[core_ext,nxh5_ext,nexus_ext,io_ext,nx_ext,
                   ex_trans_test,utils_test,numpy_utils_test],
      packages = find_packages(),
      url="https://github.com/pni-libraries/python-pni",
      test_suite="pni.test",
      test_loader = "unittest:TestLoader",
      cmdclass={"install":pni_install}
    )

