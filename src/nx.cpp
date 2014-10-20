//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pniio.
//
// python-pniio is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pniio is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
//  Created on: Jan 5, 2012
//      Author: Eugen Wintersberger
//

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <boost/python.hpp>
#include <iostream>
#include <sstream>

#include <pni/io/nx/nx.hpp>
#include <pni/io/exceptions.hpp>

using namespace pni::core;
using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni::io::nx::h5;

#include "nxgroup_wrapper.hpp"
#include "nxattribute_wrapper.hpp"
#include "nxfile_wrapper.hpp"
#include "nxfield_wrapper.hpp"
#include "child_iterator.hpp"
//#include "AttributeIterator.hpp"
#include "nxwrapper_errors.hpp"
#include "bool_converter.hpp"
#include "numpy_scalar_converter.hpp"



//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(nxh5)
{
    
    //this is absolutely necessary - otherwise the nympy API functions do not
    //work.
    import_array();

    //register converter
    bool_t_to_python_converter();
    python_to_bool_t_converter();
    numpy_scalar_converter();

    //register exception translators
    exception_registration();
   
    //wrap NX-attribute object
    wrap_nxattribute<pni::io::nx::h5::nxattribute>();
    
    //wrap NX-file
    wrap_nxfile<pni::io::nx::h5::nxfile>();
    
    //wrap NX-field
    wrap_nxfield<pni::io::nx::h5::nxfield>();
    
    //wrap NX-group
    wrap_nxgroup<pni::io::nx::h5::nxgroup>();
    wrap_childiterator<nxgroup_wrapper<pni::io::nx::h5::nxgroup>>("NXGroupChildIterator");


    //wrap NX-object
    /*
    wrap_nxobject<pni::io::nx::h5::nxobject>("NXObject");
    wrap_attributeiterator
        <NXObjectWrapper<pni::io::nx::h5::nxgroup>,
         NXAttributeWrapper<pni::io::nx::h5::nxattribute> >("NXGroupAttributeIterator");
    wrap_attributeiterator
        <NXObjectWrapper<pni::io::nx::h5::nxfield>,
         NXAttributeWrapper<pni::io::nx::h5::nxattribute> >("NXFieldAttributeIterator");
    wrap_attributeiterator
        <NXObjectWrapper<pni::io::nx::h5::nxfile>,
         NXAttributeWrapper<pni::io::nx::h5::nxattribute> >("NXFiledAttributeIterator");
         */



    //create wrapper for NXDefalteFilter
    /*
    uint32 (nxdeflate_filter::*get_compression_rate)() const =
        &nxdeflate_filter::compression_rate;
    void (nxdeflate_filter::*set_compression_rate)(uint32) =
        &nxdeflate_filter::compression_rate;
    bool (nxdeflate_filter::*get_shuffle)() const = &nxdeflate_filter::shuffle;
    void (nxdeflate_filter::*set_shuffle)(bool) = &nxdeflate_filter::shuffle;
    class_<nxdeflate_filter>("NXDeflateFilter")
        .add_property("rate",get_compression_rate,set_compression_rate)
        .add_property("shuffle",get_shuffle,set_shuffle)
        ;
        */

}
