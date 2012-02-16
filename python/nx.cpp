/*
 * nx.cpp
 *
 *  Created on: Jan 5, 2012
 *      Author: Eugen Wintersberger
 */

extern "C"{
    #include<Python.h>
}

#include <boost/python.hpp>
#include <iostream>

#include "../src/NX.hpp"
#include "../src/NXExceptions.hpp"

using namespace pni::utils;
using namespace pni::nx::h5;
using namespace boost::python;

void wrap_nxobject();
void wrap_nxgroup();
void wrap_nxfile();
void wrap_nxattribute();
void wrap_nxfield();
void wrap_nxnumericfield();
void wrap_nxstringfield();
void wrap_nxbinaryfield();

void NXFileError_translator(pni::nx::NXFileError const &error){
    PyErr_SetString(PyExc_UserWarning,"NXFileError raised by C++ code!");
}

void NXGroupError_translator(pni::nx::NXGroupError const &error){
    PyErr_SetString(PyExc_UserWarning,"NXGroupError raised by C++ code!");
}

void NXAttributeError_translator(pni::nx::NXAttributeError const &error){
    PyErr_SetString(PyExc_UserWarning,"NXAttributeError raised by C++ code!");
}

void NXFieldError_translator(pni::nx::NXFieldError const &error){
    PyErr_SetString(PyExc_UserWarning,"NXFieldError raised by C++ code!");
}

BOOST_PYTHON_MODULE(nxh5)
{
    register_exception_translator<pni::nx::NXFileError>(NXFileError_translator);
    register_exception_translator<pni::nx::NXGroupError>(NXGroupError_translator);
    register_exception_translator<pni::nx::NXAttributeError>(NXAttributeError_translator);
    register_exception_translator<pni::nx::NXFieldError>(NXFieldError_translator);

    wrap_nxobject();
    wrap_nxgroup();
    wrap_nxfile();
    wrap_nxfield();
    wrap_nxattribute();
}
