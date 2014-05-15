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

extern "C"{
#include<numpy/arrayobject.h>
}

#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>
using namespace pni::core;

#include "nxwrapper_utils.hpp"
#include "nxio_operations.hpp"
#include "numpy_utils.hpp"

//! 
//! \ingroup wrappers  
//! \brief template class to wrap attributes
//! 
//! This template provides a wrapper for attribute types.
//!
template<typename AttrType> class NXAttributeWrapper
{
    private:
        AttrType _attribute; //!< instance of the attribute type to wrap
    public:
        //====================public types=====================================
        //! wrapper type
        typedef NXAttributeWrapper<AttrType> wrapper_t;
        //===============constructors and destructor===========================
        //! default constructor
        NXAttributeWrapper():_attribute(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXAttributeWrapper(const wrapper_t &a):
            _attribute(a._attribute)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        NXAttributeWrapper(wrapper_t &&a):_attribute(std::move(a._attribute))
        {}

        //---------------------------------------------------------------------
        //! copy constructor from implementation
        explicit NXAttributeWrapper(const AttrType &a): _attribute(a)
        {}

        //---------------------------------------------------------------------
        //! move constructor from implementation
        explicit NXAttributeWrapper(AttrType &&a): _attribute(std::move(a))
        {}

        //---------------------------------------------------------------------
        //! destructor
        ~NXAttributeWrapper(){}

        //=====================assignment operator=============================
        //! copy assignment
        wrapper_t &operator=(const wrapper_t &a)
        {
            if(this != &a) _attribute = a._attribute;
            return *this;
        }

        //---------------------------------------------------------------------
        //! move assignment
        wrapper_t &operator= (wrapper_t &&a)
        {
            if(this != &a) _attribute = std::move(a._attribute);
            return *this;
        }

        //==========================inquery methodes===========================
        /*! \brief get attribute shape

        Returns the shape of an attribute as tuple. In Python shape will be a
        read only property of the attribute object. Using a tuple immediately 
        indicates that this is an immutable value. The length of the tuple is
        equal to the rank (number of dimensions) while the elements are the
        number of elements along each dimension.
        \return tuple with shape information
        */
        tuple shape() const
        {
            return tuple(Container2List(this->_attribute.template shape<shape_t>()));
        }

        //---------------------------------------------------------------------
        /*! \brief get attribute type id
        
        Returns the numpy typecode of the attribute. We do not wrapp the TypeID
        enum class to Python as this would not make too much sense. 
        However, if we use here directly the numpy codes we cann use this value
        for the instantiation of a new numpy array. 
        This value will be provided to Python users as a read-only property with
        name dtype (as in numpy).
        \return numpy typecode
        */
        string type_id() const
        {
            return numpy::type_str(this->_attribute.type_id()); 
        }

        //---------------------------------------------------------------------
        //! close the attribute
        void close() { this->_attribute.close(); }

        //----------------------------------------------------------------------
        /*! \brief query validity status

        Returns true if the attribute object is valid. The user will have access
        to this value via a read-only property names valid. 
        \return true of attribute is valid, false otherwise
        */
        bool is_valid() const { return this->_attribute.is_valid(); }

        //----------------------------------------------------------------------
        /*! \brief get attribute name

        Returns the name of the attribute object. User access is given via a
        read-only property "name". 
        \return attribute name
        */
        string name() const { return this->_attribute.name(); }

        //=========================read methods================================
        /*! \brief read attribute data

        This method reads the attributes data and returns an appropriate Python
        object holding the data. If the method is not able to decide who to
        store the data to a Python object an exception will be thrown. 
        The return value is either a simple scalar Python type or a numpy array. 
        This method is the reading part of the "value" property which provides
        access to the data of an attribuite.
        \throws nxattribute_error in case of problems
        \return Python object with attribute data
        */
        object read() const
        {
            //if(this->_attribute.template shape<shape_t>().size() == 0)
            if(this->_attribute.size() == 1)
                return io_read<scalar_reader>(this->_attribute);
            else
                return io_read<array_reader>(this->_attribute);

            //should raise an exception here
            throw pni::io::nx::nxattribute_error(EXCEPTION_RECORD,
            "Found no appropriate procedure to read this attribute!");

            //this is only to avoid compiler warnings
            return object();
        }

        //=====================write methods===================================
        /*! \brief write attribute data

        Write attribute data to disk. The data is passed as a Python object.
        The method is the writing part of the "value" property which provides
        access to the attribute data. An exception will be thrown if the method
        cannot write data from the object. For the time being the object must
        either be a numpy array or a simple Python scalar.
        \throws nxattribute_reader in case of problems
        \throws type_error if type conversion fails
        \throws shape_mismatch_error if attribute and object shape cannot be
        converted
        \param o object from which to write data
        */
        void write(object o) const
        {
            //before we can write an object we need to find out what 
            //it really i
            if(this->_attribute.template shape<shape_t>().size() == 0)
                //write to a scalar attribute
                io_write<scalar_writer>(this->_attribute,o);
            else
            {
                //the attribute is an array attribute
                if(numpy::is_array(o))
                    //write array to array
                    io_write<array_writer>(this->_attribute,o);
            }
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

/*! 
\ingroup wrappers
\brief create new NXAttribute wrapper

Template function to create a new wrapper for the NXAttribute type AType.
*/
template<typename ATYPE> void wrap_nxattribute()
{
    typedef NXAttributeWrapper<ATYPE> wrapper_t;

    class_<wrapper_t>("NXAttribute")
        .add_property("shape",&wrapper_t::shape,__attribute_shape_docstr)
        .add_property("dtype",&wrapper_t::type_id,__attribute_dtype_docstr)
        .add_property("valid",&wrapper_t::is_valid,__attribute_valid_docstr)
        .add_property("name",&wrapper_t::name,__attribute_name_docstr)
        .add_property("value",&wrapper_t::read,
                              &wrapper_t::write,__attribute_value_docstr)
        .def("close",&wrapper_t::close,__attribute_close_docstr)
        .def("write",&wrapper_t::write,__attribute_write_docstr)
        ;

}

