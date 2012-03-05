#setup script for libpninx-python package

from distutils.core import setup
from distutils.extension import Extension

libs = ["boost_python"]
files = ["src/test.cpp"]


pninx = Extension("pninx",files,libraries=libs)

setup(name="pninx",ext_modules=[pninx])

