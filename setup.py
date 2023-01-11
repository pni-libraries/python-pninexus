# setup script for python-pninexus
from __future__ import print_function
import sys
import os
import os.path
from numpy.distutils.misc_util import get_numpy_include_dirs
from setuptools import setup
from sphinx.setup_command import BuildDoc
from distutils.command.install import install
import sysconfig
from build_tools import (CppExtensionFactory,
                         ConanBuildInfoBuilder,
                         BuildConfiguration)


cmdclass = {'build_sphinx': BuildDoc}
name = "pninexus"
version = "3.0.3"
release = "3.0.3"
# release = "3.0"

if release.count(".") == 1:
    docs_release = '(latest)'
else:
    docs_release = release


def get_build_dir():
    build_dir = "lib.{platform}-{version[0]}.{version[1]}"

    return os.path.join("build",
                        build_dir.format(platform=sysconfig.get_platform(),
                                         version=sys.version_info))

# -----------------------------------------------------------------------------
# set include directories, linker path and libraries
# we have two possibilities to configure the build:
# -> via pkg-config using locally installed libraries
# -> via Conan
# if a conanbuildinfo.txt file exists in the root directory of the package the
# conan variant will be used automatically.
# -----------------------------------------------------------------------------


h5cpp_extra_link_args = []
nexus_extra_link_args = []


if os.path.exists("conanbuildinfo.txt"):
    builder = ConanBuildInfoBuilder()
    nexus_config = builder.create("conanbuildinfo.txt")
    core_config = builder.create("conanbuildinfo.txt")

    nexus_config.add_linker_argument("-Wl,-rpath,'$ORIGIN'/../../libs")

    print("linking with libraries:")
    for lib in nexus_config.link_libraries:
        print(lib)
else:
    nexus_config = BuildConfiguration()

    nexus_config.add_link_library('pninexus')
    nexus_config.add_link_library('h5cpp')
    nexus_config.add_link_library(
        "boost_python{major}{minor}".format(major=sys.version_info.major,
                                            minor=sys.version_info.minor))
    nexus_config.add_include_directory('/usr/include/hdf5/serial')

    hdf5_hl_path = os.environ.get('HDF5_HL_LOCAL_PATH')
    if hdf5_hl_path:
        # use when h5cpp compiled with --as-needed
        nexus_config.add_link_library('hdf5_hl')
        nexus_config.add_library_directory(
            # '/usr/lib/x86_64-linux-gnu/hdf5/serial/'
            hdf5_hl_path
        )

    h5cpp_path = os.environ.get('H5CPP_LOCAL_PATH')
    if h5cpp_path:
        # use when h5cpp locally installed
        nexus_config.add_include_directory("%s/include" % h5cpp_path)
        nexus_config.add_library_directory("%s/lib" % h5cpp_path)
        nexus_config.add_linker_argument("-Wl,-rpath,/%s/lib" % h5cpp_path)

    pninexus_path = os.environ.get('PNINEXUS_LOCAL_PATH')
    if pninexus_path:
        # use when libpnineuxs locally installed
        nexus_config.add_include_directory("%s/include" % pninexus_path)
        nexus_config.add_library_directory("%s/lib" % pninexus_path)
        nexus_config.add_linker_argument("-Wl,-rpath,%s/lib" % pninexus_path)

#
# adding include directories from numpy
#

nexus_config.add_include_directories(get_numpy_include_dirs())


# ----------------------------------------------------------------------------
# set compiler options
# ----------------------------------------------------------------------------
if sys.platform == "win32":
    pass
else:
    arguments = ['-std=c++11', '-Wall', '-Wextra', '-fdiagnostics-show-option']
    nexus_config.add_compiler_arguments(arguments)

# ----------------------------------------------------------------------------
# creating the extension factories
# ----------------------------------------------------------------------------
nexus_extension_factory = CppExtensionFactory(config=nexus_config)

# ----------------------------------------------------------------------------
# list of files for the pnicore extensions
# ----------------------------------------------------------------------------

core_lib_dir = os.path.join(get_build_dir(), "pninexus", "core")


# -----------------------------------------------------------------------------
# setup for the h5cpp and nexus extensions
# -----------------------------------------------------------------------------

h5cpp_common_sources = ['src/cpp/h5cpp/common/converters.cpp',
                        'src/cpp/h5cpp/numpy/dimensions.cpp',
                        'src/cpp/h5cpp/numpy/array_factory.cpp',
                        'src/cpp/h5cpp/numpy/array_adapter.cpp']

h5cpp_core_ext = nexus_extension_factory.create(
    module_name="pninexus.h5cpp._h5cpp",
    source_files=['src/cpp/h5cpp/_h5cpp.cpp',
                  'src/cpp/h5cpp/boost_filesystem_path_conversion.cpp',
                  'src/cpp/h5cpp/dimensions_conversion.cpp',
                  'src/cpp/h5cpp/errors.cpp'])

h5cpp_attribute_ext = nexus_extension_factory.create(
    module_name='pninexus.h5cpp._attribute',
    source_files=['src/cpp/h5cpp/attribute/attribute.cpp'] +
    h5cpp_common_sources)

h5cpp_file_ext = nexus_extension_factory.create(
    module_name='pninexus.h5cpp._file',
    source_files=['src/cpp/h5cpp/file/file.cpp'] +
    h5cpp_common_sources)

h5cpp_dataspace_ext = nexus_extension_factory.create(
    module_name='pninexus.h5cpp._dataspace',
    source_files=['src/cpp/h5cpp/dataspace/dataspace.cpp',
                  'src/cpp/h5cpp/dataspace/selections.cpp'])

h5cpp_datatype_ext = nexus_extension_factory.create(
    module_name='pninexus.h5cpp._datatype',
    source_files=['src/cpp/h5cpp/datatype/datatype.cpp'])

h5cpp_filter_ext = nexus_extension_factory.create(
    module_name='pninexus.h5cpp._filter',
    source_files=['src/cpp/h5cpp/filter/filter.cpp'])

h5cpp_property_ext = nexus_extension_factory.create(
    module_name='pninexus.h5cpp._property',
    source_files=['src/cpp/h5cpp/property/property.cpp',
                  'src/cpp/h5cpp/property/enumeration_wrappers.cpp',
                  'src/cpp/h5cpp/property/class_wrappers.cpp',
                  'src/cpp/h5cpp/property/copy_flag_wrapper.cpp',
                  'src/cpp/h5cpp/property/chunk_cache_parameters.cpp',
                  'src/cpp/h5cpp/property/creation_order.cpp'])

h5cpp_node_ext = nexus_extension_factory.create(
    module_name='pninexus.h5cpp._node',
    source_files=['src/cpp/h5cpp/node/nodes.cpp',
                  'src/cpp/h5cpp/node/dataset.cpp',
                  'src/cpp/h5cpp/node/functions.cpp'] + h5cpp_common_sources)

nexus_extension = nexus_extension_factory.create(
    module_name='pninexus.nexus._nexus',
    source_files=['src/cpp/nexus/nexus.cpp',
                  'src/cpp/nexus/factories.cpp',
                  'src/cpp/nexus/predicates.cpp',
                  'src/cpp/nexus/list_converters.cpp',
                  'src/cpp/nexus/path.cpp',
                  'src/cpp/nexus/element_dict_converter.cpp'])


# ----------------------------------------------------------------------------
# customized version of the `install` command. The original version shipped
# with distutils does not install header files even if they are mentioned
# in the project setup.
#
# We thus call `install_headers` manually.
# ----------------------------------------------------------------------------
class pni_install(install):
    def run(self):
        install.run(self)


# ----------------------------------------------------------------------------
# setup for the pnicore package
# ----------------------------------------------------------------------------

setup(
    name="pninexus",
    author="Eugen Wintersberger",
    author_email="eugen.wintersberger@desy.de",
    description="Python wrapper for the H5CPP and PNI libraries",
    long_description="This package provides wrappers for the PNI C++ " +
    "libraries libpninexus and libh5cpp.",
    maintainer="Eugen Wintersberger, Jan Kotanski",
    maintainer_email="jan.kotanski@desy.de",
    license="GPLv2",
    version=release,
    requires=["numpy"],
    ext_modules=[
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
    package_dir={'': 'src'},
    packages=['pninexus',
              'pninexus.h5cpp',
              'pninexus.h5cpp.attribute',
              'pninexus.h5cpp.dataspace',
              'pninexus.h5cpp.datatype',
              'pninexus.h5cpp.file',
              'pninexus.h5cpp.filter',
              'pninexus.h5cpp.node',
              'pninexus.h5cpp.property',
              'pninexus.nexus'],
    url="https://github.com/pni-libraries/python-pninexus",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite="test",
    test_loader="unittest:TestLoader",
    cmdclass={"install": pni_install},
    keywords='h5cpp hdf5 python photon science detector',
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', docs_release)
        }
    }
)
