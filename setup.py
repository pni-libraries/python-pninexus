#setup script for python-pni
from __future__ import print_function
import sys
import os
import os.path
from numpy.distutils.misc_util import get_numpy_include_dirs
from pkgconfig import package
from setuptools import setup, find_packages, Extension
from distutils.command.install import install
import sysconfig
from build_tools import CppExtensionFactory,ConanBuildInfoBuilder

def get_build_dir():
    build_dir = "lib.{platform}-{version[0]}.{version[1]}"

    return os.path.join("build",build_dir.format(platform = sysconfig.get_platform(),
                                                 version = sys.version_info))

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
h5cpp_extra_link_args =[]
nexus_extra_link_args =[]


if os.path.exists("conanbuildinfo.txt"):
    builder = ConanBuildInfoBuilder()
    nexus_config = builder.create("conanbuildinfo.txt")
    core_config  = builder.create("conanbuildinfo.txt")

    nexus_config.add_linker_argument("-Wl,-rpath,'$ORIGIN'/../../libs")
    core_config.add_linker_argument("-Wl,-rpath,'$ORIGIN'/../libs")
    
    print("linking with libraries:")
    for lib in nexus_config.link_libraries:
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


#
# adding include directories from numpy
#

nexus_config.add_include_directories(get_numpy_include_dirs())
core_config.add_include_directories(get_numpy_include_dirs())


#-----------------------------------------------------------------------------
# set compiler options
#-----------------------------------------------------------------------------
if sys.platform == "win32":
    pass
else:
    arguments = ['-std=c++11','-Wall','-Wextra','-fdiagnostics-show-option',
                 '-Wno-strict-prototypes']
    nexus_config.add_compiler_arguments(arguments)
    core_config.add_compiler_arguments(arguments)

# ----------------------------------------------------------------------------
# creating the extension factories
# ----------------------------------------------------------------------------
nexus_extension_factory = CppExtensionFactory(config = nexus_config)
core_extension_factory = CppExtensionFactory(config = core_config)


#-----------------------------------------------------------------------------
# list of files for the pnicore extensions
#-----------------------------------------------------------------------------

core_lib_dir=os.path.join(get_build_dir(),"pni","core")

#-----------------------------------------------------------------------------
# setup for the core extension
#-----------------------------------------------------------------------------
core_ext = core_extension_factory.create(
           module_name = "pni.core._core",
           source_files = ["src/core/bool_converter.cpp",
                           "src/core/numpy_scalar_converter.cpp",
                           "src/core/errors.cpp",
                           "src/core/_core.cpp",
                           "src/core/numpy_utils.cpp",
                           "src/core/utils.cpp",
                           ])

#------------------------------------------------------------------------------
# setup for the h5cpp and nexus extensions
#------------------------------------------------------------------------------

h5cpp_common_sources = ['src/io/h5cpp/common/converters.cpp',
                        'src/io/h5cpp/numpy/dimensions.cpp',
                        'src/io/h5cpp/numpy/array_factory.cpp',
                        'src/io/h5cpp/numpy/array_adapter.cpp']

h5cpp_core_ext = nexus_extension_factory.create(
                 module_name = "pni.io.h5cpp._h5cpp",
                 source_files = ['src/io/h5cpp/_h5cpp.cpp',
                                 'src/io/h5cpp/boost_filesystem_path_conversion.cpp',
                                 'src/io/h5cpp/dimensions_conversion.cpp',
                                 'src/io/h5cpp/errors.cpp'])

h5cpp_attribute_ext = nexus_extension_factory.create(
                      module_name = 'pni.io.h5cpp._attribute',
                      source_files = ['src/io/h5cpp/attribute/attribute.cpp']+h5cpp_common_sources)

h5cpp_file_ext = nexus_extension_factory.create(
                      module_name = 'pni.io.h5cpp._file',
                      source_files = ['src/io/h5cpp/file/file.cpp'])

h5cpp_dataspace_ext = nexus_extension_factory.create(
                      module_name = 'pni.io.h5cpp._dataspace',
                      source_files = ['src/io/h5cpp/dataspace/dataspace.cpp',
                                      'src/io/h5cpp/dataspace/selections.cpp'])

h5cpp_datatype_ext = nexus_extension_factory.create(
                     module_name = 'pni.io.h5cpp._datatype',
                     source_files = ['src/io/h5cpp/datatype/datatype.cpp'])

h5cpp_filter_ext = nexus_extension_factory.create(
                   module_name = 'pni.io.h5cpp._filter',
                   source_files = ['src/io/h5cpp/filter/filter.cpp'])

h5cpp_property_ext = nexus_extension_factory.create(
                     module_name = 'pni.io.h5cpp._property',
                     source_files = ['src/io/h5cpp/property/property.cpp',
                                     'src/io/h5cpp/property/enumeration_wrappers.cpp',
                                     'src/io/h5cpp/property/class_wrappers.cpp',
                                     'src/io/h5cpp/property/copy_flag_wrapper.cpp',
                                     'src/io/h5cpp/property/chunk_cache_parameters.cpp',
                                     'src/io/h5cpp/property/creation_order.cpp'])

h5cpp_node_ext = nexus_extension_factory.create(
                 module_name = 'pni.io.h5cpp._node',
                 source_files = ['src/io/h5cpp/node/nodes.cpp',
                                 'src/io/h5cpp/node/dataset.cpp',
                                 'src/io/h5cpp/node/functions.cpp',]+h5cpp_common_sources)

nexus_extension = nexus_extension_factory.create(
                  module_name = 'pni.io.nexus._nexus',
                  source_files = ['src/io/nexus/nexus.cpp',
                                  'src/io/nexus/factories.cpp',
                                  'src/io/nexus/predicates.cpp',
                                  'src/io/nexus/list_converters.cpp',
                                  'src/io/nexus/path.cpp',])




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
      ext_modules=[core_ext,
                   h5cpp_core_ext,
                   h5cpp_attribute_ext,
                   h5cpp_file_ext,
                   h5cpp_dataspace_ext,
                   h5cpp_datatype_ext,
                   h5cpp_property_ext,
                   h5cpp_filter_ext,
                   h5cpp_node_ext,
                   nexus_extension,
                   ],
      packages = find_packages(),
      url="https://github.com/pni-libraries/python-pni",
      test_suite="test",
      test_loader = "unittest:TestLoader",
      cmdclass={"install":pni_install}
    )

