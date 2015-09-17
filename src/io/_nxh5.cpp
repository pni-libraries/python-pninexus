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


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <boost/python.hpp>
#include <iostream>
#include <sstream>

#include <pni/io/nx/nx.hpp>
#include <pni/io/exceptions.hpp>


//import here the namespace for the nxh5 module
using namespace pni::core;
using namespace boost::python;
using namespace pni::io::nx;

#include "nxgroup_wrapper.hpp"
#include "nxattribute_wrapper.hpp"
#include "nxfile_wrapper.hpp"
#include "nxfield_wrapper.hpp"
#include "child_iterator.hpp"
#include "nxobject_to_python_converter.hpp"
#include "nxgroup_to_python_converter.hpp"
#include "nxfield_to_python_converter.hpp"
#include "nxattribute_to_python_converter.hpp"
#include "nxattribute_manager_wrapper.hpp"
#include "xml_functions_wrapper.hpp"
//#include "algorithms_wrapper.hpp"


#if PY_MAJOR_VERSION >= 3
int
#else 
void
#endif
init_numpy()
{
    import_array();
}

extern void create_nxattribute_wrappers();

//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(_nxh5)
{
    typedef nxobject_to_python_converter<h5::nxobject,
                                         h5::nxgroup,
                                         h5::nxfield,
                                         h5::nxattribute> object_converter_type;
    typedef nxgroup_to_python_converter<h5::nxgroup> group_converter_type;
    typedef nxfield_to_python_converter<h5::nxfield> field_converter_type;
    typedef nxattribute_to_python_converter<h5::nxattribute>
        attribute_converter_type;
    //this is absolutely necessary - otherwise the nympy API functions do not
    //work.
    init_numpy();

    //register converter
    to_python_converter<h5::nxobject,object_converter_type>();
    to_python_converter<h5::nxgroup,group_converter_type>();
    to_python_converter<h5::nxfield,field_converter_type>();
    to_python_converter<h5::nxattribute,attribute_converter_type>();

    //wrap NX-attribute object
    //wrap_nxattribute<h5::nxattribute>();
    create_nxattribute_wrappers();
    
    //wrap NX-file
    wrap_nxfile<h5::nxfile>();
    
    //wrap NX-field
    wrap_nxfield<h5::nxfield>();
    wrap_nxattribute_manager<decltype(h5::nxfield::attributes)>("nxfield_attributes");
    
    //wrap NX-group
    wrap_nxgroup<h5::nxgroup>();
    wrap_childiterator<nxgroup_wrapper<h5::nxgroup>>("NXGroupChildIterator");
    wrap_nxattribute_manager<decltype(h5::nxgroup::attributes)>("nxgroup_attributes");

    //create the XML function wrappers
    create_xml_function_wrappers<h5::nxgroup>();
    //create_algorithms_wrappers<h5::nxobject>();

    //create wrapper for NXDefalteFilter
    uint32 (h5::nxdeflate_filter::*get_compression_rate)() const =
           &h5::nxdeflate_filter::compression_rate;
    void (h5::nxdeflate_filter::*set_compression_rate)(uint32) =
          &h5::nxdeflate_filter::compression_rate;
    bool (h5::nxdeflate_filter::*get_shuffle)() const =
        &h5::nxdeflate_filter::shuffle;
    void (h5::nxdeflate_filter::*set_shuffle)(bool) =
        &h5::nxdeflate_filter::shuffle;
    class_<h5::nxdeflate_filter>("deflate_filter")
        .add_property("rate",get_compression_rate,set_compression_rate)
        .add_property("shuffle",get_shuffle,set_shuffle)
        ;

}
