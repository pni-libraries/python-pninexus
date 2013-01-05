/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of python-pniio.
 *
 * python-pniio is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * python-pniio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
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

#include <pni/io/nx/NX.hpp>
#include <pni/io/nx/NXExceptions.hpp>

using namespace pni::core;
using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni::io::nx::h5;

#include "NXObjectWrapper.hpp"
#include "NXGroupWrapper.hpp"
#include "NXAttributeWrapper.hpp"
#include "NXFileWrapper.hpp"
#include "NXObjectMap.hpp"
#include "NXFieldWrapper.hpp"
#include "ChildIterator.hpp"
#include "AttributeIterator.hpp"
#include "NXWrapperErrors.hpp"



//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(nxh5)
{
    
    //this is absolutely necessary - otherwise the nympy API functions do not
    //work.
    import_array();


    //register exception translators
    exception_registration();
   
    //wrap NX-attribute object
    wrap_nxattribute<pni::io::nx::h5::NXAttribute>();

    //wrap NX-object
    wrap_nxobject<pni::io::nx::h5::NXObject>("NXObject");
    wrap_attributeiterator
        <NXObjectWrapper<pni::io::nx::h5::NXGroup>,
         NXAttributeWrapper<pni::io::nx::h5::NXAttribute> >("NXGroupAttributeIterator");
    wrap_attributeiterator
        <NXObjectWrapper<pni::io::nx::h5::NXField>,
         NXAttributeWrapper<pni::io::nx::h5::NXAttribute> >("NXFieldAttributeIterator");
    wrap_attributeiterator
        <NXObjectWrapper<pni::io::nx::h5::NXFile>,
         NXAttributeWrapper<pni::io::nx::h5::NXAttribute> >("NXFiledAttributeIterator");

    //wrap NX-group
    wrap_nxobject<pni::io::nx::h5::NXGroup>("NXObject_GroupInstance");
    wrap_nxgroup<pni::io::nx::h5::NXGroup>("NXGroup");
    wrap_childiterator<NXGroupWrapper<pni::io::nx::h5::NXGroup>
        >("NXGroupChildIterator");

    //wrap NX-field
    wrap_nxobject<pni::io::nx::h5::NXField>("NXObject_FieldInstance");
    wrap_nxfield<pni::io::nx::h5::NXField>("NXField");

    //wrap NX-file
    wrap_nxobject<pni::io::nx::h5::NXFile>("NXObject_FileInstance");
    wrap_nxgroup<pni::io::nx::h5::NXFile>("NXGroup_FileInstance");
    wrap_nxfile<pni::io::nx::h5::NXFile>("NXFile");
    wrap_childiterator<NXGroupWrapper<pni::io::nx::h5::NXFile>
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
