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

#include <pni/nx/NX.hpp>
#include <pni/nx/NXExceptions.hpp>

using namespace pni::utils;
using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni::nx::h5;

#include "NXObjectWrapper.hpp"
#include "NXGroupWrapper.hpp"
#include "NXAttributeWrapper.hpp"
#include "NXFileWrapper.hpp"
#include "NXObjectMap.hpp"


template<> class NXObjectMap<pni::nx::h5::NXGroup>{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXSelection SelectionType;
};

template<> class NXObjectMap<pni::nx::h5::NXFile>{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXSelection SelectionType;
};
/*
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
}*/

BOOST_PYTHON_MODULE(nxh5)
{
    /*
    register_exception_translator<pni::nx::NXFileError>(NXFileError_translator);
    register_exception_translator<pni::nx::NXGroupError>(NXGroupError_translator);
    register_exception_translator<pni::nx::NXAttributeError>(NXAttributeError_translator);
    register_exception_translator<pni::nx::NXFieldError>(NXFieldError_translator);
    */
    wrap_nxobject<pni::nx::h5::NXObject>("NXObject");

    wrap_nxobject<pni::nx::h5::NXGroup>("NXObject_GroupInstance");
    wrap_nxgroup<pni::nx::h5::NXGroup>("NXGroup");

    wrap_nxattribute<pni::nx::h5::NXAttribute>();

    wrap_nxobject<pni::nx::h5::NXFile>("NXObject_FileInstance");
    wrap_nxgroup<pni::nx::h5::NXFile>("NXGroup_FileInstance");
    wrap_nxfile<pni::nx::h5::NXFile>("NXFile");

}
