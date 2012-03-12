/*
 * nx.cpp
 *
 *  Created on: Jan 5, 2012
 *      Author: Eugen Wintersberger
 */

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <boost/python.hpp>
#include <iostream>
#include <sstream>

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
#include "NXFieldWrapper.hpp"

template<> class NXObjectMap<pni::nx::h5::NXObject>{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXSelection SelectionType;
        typedef pni::nx::h5::NXAttribute AttributeType;
};

template<> class NXObjectMap<pni::nx::h5::NXGroup>{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXSelection SelectionType;
        typedef pni::nx::h5::NXAttribute AttributeType;
};

template<> class NXObjectMap<pni::nx::h5::NXField>{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXSelection SelectionType;
        typedef pni::nx::h5::NXAttribute AttributeType;
};
template<> class NXObjectMap<pni::nx::h5::NXFile>{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXSelection SelectionType;
        typedef pni::nx::h5::NXAttribute AttributeType;
};

//===============exception translators=========================================
void NXFileError_translator(pni::nx::NXFileError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void NXGroupError_translator(pni::nx::NXGroupError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void NXAttributeError_translator(pni::nx::NXAttributeError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void NXFieldError_translator(pni::nx::NXFieldError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

void NXSelectionError_translator(pni::nx::NXSelectionError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void IndexError_translator(pni::utils::IndexError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_IndexError,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void MemoryAccessError_translator(pni::utils::MemoryAccessError const &error)
{
    std::stringstream estr;
    estr << error;
    PyErr_SetString(PyExc_MemoryError,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void MemoryAllocationError_translator(pni::utils::MemoryAllocationError const
        &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_MemoryError,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void SizeMissmatchError_translator(pni::utils::SizeMissmatchError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_IndexError,error.description().c_str());
}

//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(nxh5)
{
    register_exception_translator<pni::nx::NXFileError>
        (NXFileError_translator);
    register_exception_translator<pni::nx::NXGroupError>
        (NXGroupError_translator);
    register_exception_translator<pni::nx::NXAttributeError>
        (NXAttributeError_translator);
    register_exception_translator<pni::nx::NXFieldError>
        (NXFieldError_translator);
    register_exception_translator<pni::utils::IndexError>
        (IndexError_translator);
    register_exception_translator<pni::utils::MemoryAccessError>
        (MemoryAccessError_translator);
    register_exception_translator<pni::utils::MemoryAllocationError>
        (MemoryAllocationError_translator);
    register_exception_translator<pni::utils::SizeMissmatchError>
        (SizeMissmatchError_translator);
    register_exception_translator<pni::nx::NXSelectionError>
        (NXSelectionError_translator);

    //this is absolutely necessary - otherwise the nympy API functions do not
    //work.
    import_array();
   
    //wrap NX-attribute object
    wrap_nxattribute<pni::nx::h5::NXAttribute>();

    //wrap NX-object
    wrap_nxobject<pni::nx::h5::NXObject>("NXObject");

    //wrap NX-group
    wrap_nxobject<pni::nx::h5::NXGroup>("NXObject_GroupInstance");
    wrap_nxgroup<pni::nx::h5::NXGroup>("NXGroup");

    //wrap NX-field
    wrap_nxobject<pni::nx::h5::NXField>("NXObject_FieldInstance");
    wrap_nxfield<pni::nx::h5::NXField>("NXField");

    //wrap NX-file
    wrap_nxobject<pni::nx::h5::NXFile>("NXObject_FileInstance");
    wrap_nxgroup<pni::nx::h5::NXFile>("NXGroup_FileInstance");
    wrap_nxfile<pni::nx::h5::NXFile>("NXFile");

}
