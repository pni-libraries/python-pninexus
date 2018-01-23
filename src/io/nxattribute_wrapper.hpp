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

#include <pni/io/nexus.hpp>
#include <boost/python.hpp>
#include <pni/core/error.hpp>
#include <pni/core/types.hpp>
#include <core/utils.hpp>
#include <core/numpy_utils.hpp>
#include <pni/io/exceptions.hpp>
#include <pni/io/nx/algorithms/get_path.hpp>

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
class AttributeWrapper
{
  private:
    //!
    //! instance of the attribute type to wrap
    //!
    hdf5::attribute::Attribute attribute_;
  public:
    //===============constructors and destructor===========================
    //!
    //! default constructor
    //!
    AttributeWrapper() = default;

    //!
    //! copy constructor
    //!
    AttributeWrapper(const AttributeWrapper &) = default;

    //!
    //! move constructor
    //!
    AttributeWrapper(AttributeWrapper &&) = default;

    //!
    //! copy constructor from implementation
    //!
    explicit AttributeWrapper(const hdf5::attribute::Attribute &a);

    //!
    //! move constructor from implementation
    //!
    explicit AttributeWrapper(hdf5::attribute::Attribute &&a);

    operator hdf5::attribute::Attribute() const
    {
      return attribute_;
    }

    //!
    //! @brief get attribute shape
    //!
    //! Returns the shape of an attribute as tuple. In Python shape will
    //! be a read only property of the attribute object. Using a tuple
    //! immediately indicates that this is an immutable value. The length
    //! of the tuple is equal to the rank (number of dimensions) while
    //! the elements are the number of elements along each dimension.
    //!
    //! @return tuple with shape information
    //!
    boost::python::tuple shape() const;

    //!
    //! @brief get attribute type id
    //!
    //! Returns the numpy typecode of the attribute. We do not wrapp the
    //! type_id_t enum class to Python as this would not make too much
    //! sense.  However, if we use here directly the numpy codes we cann
    //! use this value for the instantiation of a new numpy array.
    //! This value will be provided to Python users as a read-only property
    //! with name dtype (as in numpy).
    //!
    //! @return numpy typecode
    //!
    pni::core::string type_id() const;


    //!
    //! close the attribute
    //!
    void close();

    //!
    //! @brief query validity status
    //!
    //! Returns true if the attribute object is valid. The user will have
    //! access to this value via a read-only property names valid.
    //!
    //! @return true of attribute is valid, false otherwise
    //!
    bool is_valid() const;

    //!
    //! @brief get attribute name
    //!
    //! Returns the name of the attribute object. User access is given
    //! via a read-only property "name".
    //!
    //! @return attribute name
    //!
    std::string name() const;

    //!
    //! @brief read attribute data
    //!
    //! This method reads the attributes data and returns an appropriate
    //! Python object holding the data. If the method is not able to decide
    //! who to store the data to a Python object an exception will be
    //! thrown.
    //! The return value is either a simple scalar Python type or a numpy
    //! array.  This method is the reading part of the "value" property
    //! which provides access to the data of an attribuite.
    //!
    //! @throws nxattribute_error in case of problems
    //! @return Python object with attribute data
    //!
    boost::python::object read() const;

    //!
    //! @brief write attribute data
    //!
    //! Write attribute data to disk. The data is passed as a Python object.
    //! The method is the writing part of the "value" property which
    //! provides access to the attribute data. An exception will be
    //! thrown if the method cannot write data from the object. For the
    //! time being the object must either be a numpy array or a simple
    //! Python scalar.
    //!
    //! @throws nxattribute_reader in case of problems
    //! @throws type_error if type conversion fails
    //! @throws shape_mismatch_error if attribute and object shape cannot be
    //! converted
    //! @param o object from which to write data
    //!
    void write(const boost::python::object &o) const;

    //!
    //! @brief the core __getitem__ implementation
    //!
    //! The most fundamental implementation of the __getitem__ method.
    //! The tuple passed to the method can contain indices, slices, and a
    //! single ellipsis. This method is doing the real work - all other
    //! __getitem__ implementations simply call this method.
    //!
    //! @param t tuple with
    //!
    boost::python::object __getitem__(const boost::python::object &t);

    //!
    //! @brief __setitem__ implementation
    //!
    //! As for __getitem__ this method is called if a user invokes the
    //! __setitem__ method on an NXField object in Python. The method
    //! converts the object passed as input argument to a tuple if
    //! necessary and then moves on to __setitem__tuple.
    //!
    //! @param o selection object
    //! @param d Python object holding the data
    //!
    void __setitem__(const boost::python::object &t,
                     const boost::python::object &o);

    //--------------------------------------------------------------------
    std::string path() const;

    //---------------------------------------------------------------------
    size_t size() const;

    //---------------------------------------------------------------------'
    boost::python::object parent() const;

    std::string filename() const;

};


//!
//! \brief convert nattribute instances to their wrapper type
//!
//! Converts instances of nxattribute to their nxattribute_wrapper counterpart
//! on the python side.
//!
//! \tparam ATYPE attribute type
//!
struct AttributeToPythonObject
{
    //------------------------------------------------------------------------
    //!
    //! \brief perform the conversion
    //!
    static PyObject *convert(const hdf5::attribute::Attribute &attribute);

};



//! 
//! \ingroup wrappers
//! \brief create new NXAttribute wrapper
//! 
//! Template function to create a new wrapper for the NXAttribute type 
//! AType.
//!
void wrap_nxattribute(const char *class_name);

