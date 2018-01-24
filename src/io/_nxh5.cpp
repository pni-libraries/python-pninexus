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
#include <boost/python/docstring_options.hpp>
#include <iostream>
#include <sstream>

#include <pni/io/nexus.hpp>
#include <pni/io/exceptions.hpp>


//import here the namespace for the nxh5 module
using namespace boost::python;

//#include "nxobject_to_python_converter.hpp"
//#include "nxgroup_to_python_converter.hpp"

#include "iterator_wrapper.hpp"
#include "nxattribute_wrapper.hpp"
#include "nxgroup_wrapper.hpp"
#include "nxattribute_manager_wrapper.hpp"
#include "nxfile_wrapper.hpp"
#include "node_to_python.hpp"

//#include "nxfield_to_python_converter.hpp"

//#include "algorithms_wrapper.hpp"
//#include "nxlink_wrapper.hpp"


#if PY_MAJOR_VERSION >= 3
int
#else 
void
#endif
init_numpy()
{
    import_array();
}

static const pni::core::string nxdeflate_rate_doc = 
"read/write property to set and get the compression rate as an integer "
"between 0 and 9\n";

static const pni::core::string nxdeflate_shuffle_doc = 
"read/write boolean property to switch shuffeling on and of "
"(:py:const:`True` or :py:const:`False`)\n";

//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(_nxh5)
{
//    typedef nxobject_to_python_converter<h5::nxobject,
//                                         h5::nxgroup,
//                                         h5::nxfield,
//                                         h5::nxattribute,
//                                         h5::nxlink> object_converter_type;
//    typedef nxgroup_to_python_converter<h5::nxgroup> group_converter_type;
//    typedef nxfield_to_python_converter<h5::nxfield> field_converter_type;
//    typedef nxattribute_to_python_converter<h5::nxattribute>
//        attribute_converter_type;
    //this is absolutely necessary - otherwise the nympy API functions do not
    //work.
    init_numpy();

    docstring_options doc_opts; 
    doc_opts.disable_signatures();
    doc_opts.enable_user_defined();

    // ======================================================================
    // Register object converters
    // ======================================================================
    to_python_converter<hdf5::node::Node,NodeToPython>();
    to_python_converter<hdf5::node::Group,GroupToPythonObject>();
    //to_python_converter<hdf5::node::Dataset,DatasetToPythonObject>();
    to_python_converter<hdf5::attribute::Attribute,AttributeToPythonObject>();

    //wrap_link<h5::nxgroup,h5::nxfield>();
    //wrap_nxlink<nximp_code::HDF5>();


    // ======================================================================
    // Register iterator wrappers
    // ======================================================================
    nexus::register_iterator_wrapper<nexus::NodeIteratorWrapper>("node_iterator");
    nexus::register_iterator_wrapper<nexus::RecursiveNodeIteratorWrapper>("rec_node_iterator");
    nexus::register_iterator_wrapper<nexus::LinkIteratorWrapper>("link_iterator");
    nexus::register_iterator_wrapper<nexus::RecursiveLinkIteratorWrapper>("rec_link_iterator");
    nexus::register_iterator_wrapper<nexus::AttributeIteratorWrapper>("attribute_iterator");

    // =======================================================================
    // Register object wrappers
    // =======================================================================
    wrap_nxattribute_manager("nxattribute_manager");
    wrap_nxattribute("nxattribute");
    wrap_nxgroup("nxgroup");
    wrap_nxfile("nxfile");
    //wrap_nxfield("nxfield");
    

    //create the XML function wrappers
    //create_xml_wrappers();
    //create_algorithms_wrappers<h5::nxgroup,h5::nxfield,h5::nxattribute>();

    //create wrapper for NXDefalteFilter
//    uint32 (h5::nxdeflate_filter::*get_compression_rate)() const =
//           &h5::nxdeflate_filter::compression_rate;
//    void (h5::nxdeflate_filter::*set_compression_rate)(uint32) =
//          &h5::nxdeflate_filter::compression_rate;
//    bool (h5::nxdeflate_filter::*get_shuffle)() const =
//        &h5::nxdeflate_filter::shuffle;
//    void (h5::nxdeflate_filter::*set_shuffle)(bool) =
//        &h5::nxdeflate_filter::shuffle;
//    class_<h5::nxdeflate_filter>("deflate_filter")
//        .add_property("rate",get_compression_rate,set_compression_rate,nxdeflate_rate_doc.c_str())
//        .add_property("shuffle",get_shuffle,set_shuffle,nxdeflate_shuffle_doc.c_str())
//        ;

}
