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
// Created on: Feb 17, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <pni/io/nx/nx.hpp>
#include <boost/python.hpp>
#include <pni/core/error.hpp>
#include <pni/core/types.hpp>
#include <core/utils.hpp>
#include <core/numpy_utils.hpp>
#include <pni/io/exceptions.hpp>
#include <pni/io/nx/algorithms/get_path.hpp>

#include "scalar_reader.hpp"
#include "scalar_writer.hpp"
#include "array_reader.hpp"
#include "array_writer.hpp"
#include "nxio_operations.hpp"
#include "utils.hpp"
#include "io_operations.hpp"

//! 
//! \ingroup wrappers  
//! \brief template class to wrap attributes
//! 
//! This template provides a wrapper for attribute types.
//! 
//! \tparam ATYPE C++ attribute type
//!
template<typename ATYPE> class nxattribute_wrapper
{
    public:
        typedef ATYPE attribute_type;
        typedef nxattribute_wrapper<attribute_type> wrapper_type;
    private:
        attribute_type _attribute; //!< instance of the attribute type to wrap
    public:
        //====================public types=====================================
        //! wrapper type
        //===============constructors and destructor===========================
        //! default constructor
        nxattribute_wrapper():_attribute(){}

        //---------------------------------------------------------------------
        //! copy constructor
        nxattribute_wrapper(const wrapper_type &a):
            _attribute(a._attribute)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        nxattribute_wrapper(wrapper_type &&a):
            _attribute(std::move(a._attribute))
        {}

        //---------------------------------------------------------------------
        //! copy constructor from implementation
        explicit nxattribute_wrapper(const attribute_type &a): 
            _attribute(a)
        {}

        //---------------------------------------------------------------------
        //! move constructor from implementation
        explicit nxattribute_wrapper(attribute_type &&a): 
            _attribute(std::move(a))
        {}

        operator attribute_type() const 
        {
            return _attribute;
        }

        //==========================inquery methodes===========================
        //!
        //! \brief get attribute shape
        //!
        //! Returns the shape of an attribute as tuple. In Python shape will 
        //! be a read only property of the attribute object. Using a tuple 
        //! immediately indicates that this is an immutable value. The length 
        //! of the tuple is equal to the rank (number of dimensions) while 
        //! the elements are the number of elements along each dimension.
        //!
        //! \return tuple with shape information
        //!
        boost::python::tuple shape() const
        {
            auto shape = _attribute.template shape<pni::core::shape_t>();
            return boost::python::tuple(Container2List(shape));
        }

        //---------------------------------------------------------------------
        //!
        //! \brief get attribute type id
        //! 
        //! Returns the numpy typecode of the attribute. We do not wrapp the 
        //! type_id_t enum class to Python as this would not make too much 
        //! sense.  However, if we use here directly the numpy codes we cann 
        //! use this value for the instantiation of a new numpy array. 
        //! This value will be provided to Python users as a read-only property 
        //! with name dtype (as in numpy).
        //!
        //! \return numpy typecode
        //!
        pni::core::string type_id() const
        {
            return numpy::type_str(_attribute.type_id()); 
        }

        //---------------------------------------------------------------------
        //! close the attribute
        void close() { _attribute.close(); }

        //----------------------------------------------------------------------
        //!
        //! \brief query validity status
        //!
        //! Returns true if the attribute object is valid. The user will have 
        //! access to this value via a read-only property names valid. 
        //!
        //! \return true of attribute is valid, false otherwise
        //!
        bool is_valid() const { return _attribute.is_valid(); }

        //----------------------------------------------------------------------
        //!
        //! \brief get attribute name
        //! 
        //! Returns the name of the attribute object. User access is given 
        //! via a read-only property "name". 
        //!
        //! \return attribute name
        //!
        pni::core::string name() const { return _attribute.name(); }

        //=========================read methods================================
        //!
        //! \brief read attribute data
        //! 
        //! This method reads the attributes data and returns an appropriate 
        //! Python object holding the data. If the method is not able to decide 
        //! who to store the data to a Python object an exception will be 
        //! thrown. 
        //! The return value is either a simple scalar Python type or a numpy 
        //! array.  This method is the reading part of the "value" property 
        //! which provides access to the data of an attribuite.
        //!
        //! \throws nxattribute_error in case of problems
        //! \return Python object with attribute data
        //!
        boost::python::object read() const
        {
            using namespace pni::core;
            using namespace boost::python;

            object np_array = read_data(_attribute);

            if(numpy::get_size(np_array)==1)
                np_array = get_first_element(np_array);

            return np_array;
        }

        //=====================write methods===================================
        //!
        //! \brief write attribute data
        //!
        //! Write attribute data to disk. The data is passed as a Python object.
        //! The method is the writing part of the "value" property which 
        //! provides access to the attribute data. An exception will be 
        //! thrown if the method cannot write data from the object. For the 
        //! time being the object must either be a numpy array or a simple 
        //! Python scalar.
        //!
        //! \throws nxattribute_reader in case of problems
        //! \throws type_error if type conversion fails
        //! \throws shape_mismatch_error if attribute and object shape cannot be
        //! converted
        //! \param o object from which to write data
        //!
        void write(boost::python::object o) const
        {
            using namespace pni::core;

            if(numpy::is_array(o))
                write_data(_attribute,o);
            else
                write_data(_attribute,numpy::to_numpy_array(o));
        }

        //---------------------------------------------------------------------
        //!
        //! \brief the core __getitem__ implementation
        //!
        //! The most fundamental implementation of the __getitem__ method. 
        //! The tuple passed to the method can contain indices, slices, and a 
        //! single ellipsis. This method is doing the real work - all other 
        //! __getitem__ implementations simply call this method.
        //!
        //! \param t tuple with 
        //!
        boost::python::object __getitem__(const boost::python::object &t)
        {
            using namespace pni::core;
            using namespace boost::python;

            typedef std::vector<pni::core::slice> selection_type;

            tuple sel; 
            if(PyTuple_Check(t.ptr()))
                sel = tuple(t);
            else
                sel = make_tuple<object>(t);

            //first we need to create a selection
            selection_type selection = create_selection(sel,_attribute);
            
            object np_array = read_data(_attribute(selection));

            if(numpy::get_size(np_array)==1) 
                np_array = get_first_element(np_array);

            return np_array;
        }

        //---------------------------------------------------------------------
        //!
        //! \brief __setitem__ implementation
        //!
        //! As for __getitem__ this method is called if a user invokes the
        //! __setitem__ method on an NXField object in Python. The method 
        //! converts the object passed as input argument to a tuple if 
        //! necessary and then moves on to __setitem__tuple.
        //!
        //! \param o selection object
        //! \param d Python object holding the data
        //!
        void __setitem__(const boost::python::object &t,
                         const boost::python::object &o)
        {
            using namespace pni::core;
            using namespace boost::python;

            typedef std::vector<pni::core::slice> selection_type;
            tuple sel;

            if(PyTuple_Check(t.ptr()))
                sel = tuple(t);
            else
                sel = make_tuple<object>(t);

            selection_type selection = create_selection(sel,_attribute);

            if(numpy::is_array(o))
                write_data(_attribute(selection),o);
            else
                write_data(_attribute(selection),numpy::to_numpy_array(o));

        }

        //--------------------------------------------------------------------
        pni::core::string path() const
        {
            return pni::io::nx::get_path(_attribute);
        }

        //---------------------------------------------------------------------
        size_t size() const
        {
            return _attribute.size();
        }
        
        //---------------------------------------------------------------------
        pni::core::string filename() const
        {
            return _attribute.filename();
        }

        //---------------------------------------------------------------------'
        auto parent() const -> decltype(_attribute.parent())
        {
            return _attribute.parent();
        }

};

//=============================================================================
static const char __attribute_shape_docstr[] =
"Read only property providing the shape of the attribute as tuple.\n"
"The length of the tuple corresponds to the number of dimensions of the\n"
"attribute and its elements denote the number of elements along each\n"
"of these dimensions.";
static const char __attribute_dtype_docstr[] =
"Read only property providing the data-type of the attribute as numpy\n"
"type-code";
static const char __attribute_valid_docstr[] =
"Read only property with a boolean value. If true the attribute is\n"
"valid. If false the object became invalid and no data can be\n"
"read or writen from and to it.";
static const char __attribute_name_docstr[] = 
"A read only property providing the name of the attribute as a string.";
static const char __attribute_value_docstr[] = 
"Read/write property to read and write attribute data.";
static const char __attribute_close_docstr[] = 
"Class method to close an open attribute. Although, attributes are \n"
"closed automatically when they are no longer referenced. This method\n"
"can be used to force the closeing an attribute.";

static const char __attribute_write_docstr[] = 
"Write all the data of an attribute at once. The argument passed to this\n"
"function is either a single scalar object or an instance of a numpy\n"
"array.";

//! 
//! \ingroup wrappers
//! \brief create new NXAttribute wrapper
//! 
//! Template function to create a new wrapper for the NXAttribute type 
//! AType.
//!
template<typename ATYPE> void wrap_nxattribute()
{
    using namespace boost::python;

    typedef nxattribute_wrapper<ATYPE> wrapper_type;

    class_<wrapper_type>("nxattribute")
        .add_property("dtype",&wrapper_type::type_id,__attribute_dtype_docstr)
        .add_property("shape",&wrapper_type::shape,__attribute_shape_docstr)
        .add_property("size",&wrapper_type::size)
        .add_property("filename",&wrapper_type::filename)
        .add_property("name",&wrapper_type::name,__attribute_name_docstr)
        .add_property("parent",&wrapper_type::parent)
        .add_property("is_valid",&wrapper_type::is_valid,__attribute_valid_docstr)
        .add_property("path",&wrapper_type::path)
        .def("close",&wrapper_type::close,__attribute_close_docstr)
        .def("read",&wrapper_type::read)
        .def("write",&wrapper_type::write,__attribute_write_docstr)
        .def("__getitem__",&wrapper_type::__getitem__)
        .def("__setitem__",&wrapper_type::__setitem__)
        ;

}

