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
#include "ChildIterator.hpp"
#include "AttributeIterator.hpp"
#include "NXWrapperErrors.hpp"

//! \cond NO_API_DOC
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

//! \endcond


//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(nxh5)
{
    
    //this is absolutely necessary - otherwise the nympy API functions do not
    //work.
    import_array();


    //register exception translators
    exception_registration();
   
    //wrap NX-attribute object
    wrap_nxattribute<pni::nx::h5::NXAttribute>();

    //wrap NX-object
    wrap_nxobject<pni::nx::h5::NXObject>("NXObject");
    wrap_attributeiterator
        <NXObjectWrapper<pni::nx::h5::NXGroup>,
         NXAttributeWrapper<pni::nx::h5::NXAttribute> >("NXGroupAttributeIterator");
    wrap_attributeiterator
        <NXObjectWrapper<pni::nx::h5::NXField>,
         NXAttributeWrapper<pni::nx::h5::NXAttribute> >("NXFieldAttributeIterator");
    wrap_attributeiterator
        <NXObjectWrapper<pni::nx::h5::NXFile>,
         NXAttributeWrapper<pni::nx::h5::NXAttribute> >("NXFiledAttributeIterator");

    //wrap NX-group
    wrap_nxobject<pni::nx::h5::NXGroup>("NXObject_GroupInstance");
    wrap_nxgroup<pni::nx::h5::NXGroup>("NXGroup");
    wrap_childiterator<NXGroupWrapper<pni::nx::h5::NXGroup>
        >("NXGroupChildIterator");

    //wrap NX-field
    wrap_nxobject<pni::nx::h5::NXField>("NXObject_FieldInstance");
    wrap_nxfield<pni::nx::h5::NXField>("NXField");

    //wrap NX-file
    wrap_nxobject<pni::nx::h5::NXFile>("NXObject_FileInstance");
    wrap_nxgroup<pni::nx::h5::NXFile>("NXGroup_FileInstance");
    wrap_nxfile<pni::nx::h5::NXFile>("NXFile");
    wrap_childiterator<NXGroupWrapper<pni::nx::h5::NXFile>
        >("NXFileChildIterator");

    //create wrapper for NXDefalteFilter

    UInt32 (NXDeflateFilter::*get_compression_rate)() const =
        &NXDeflateFilter::compression_rate;
    void (NXDeflateFilter::*set_compression_rate)(UInt32) =
        &NXDeflateFilter::compression_rate;
    bool (NXDeflateFilter::*get_shuffle)() const = &NXDeflateFilter::shuffle;
    void (NXDeflateFilter::*set_shuffle)(bool) = &NXDeflateFilter::shuffle;
    class_<NXDeflateFilter>("NXDeflateFilter")
        .add_property("rate",get_compression_rate,set_compression_rate)
        .add_property("shuffle",get_shuffle,set_shuffle)
        ;

}
