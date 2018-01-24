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
// along with pyton-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Feb 17, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#pragma once

#include <boost/python.hpp>
#include <pni/io/nexus.hpp>

#include "nxattribute_manager_wrapper.hpp"
#include "iterator_wrapper.hpp"

//! 
//! \ingroup wrappers  
//! \brief class tempalte for NXGroup wrapper
//! 
//! Class template to create wrappers for NXGroup types.
//!
class GroupWrapper
{
  private:
    hdf5::node::Group group_;

    static hdf5::datatype::Datatype create_datatype(const std::string &);
  public:
    //================constructors and destructor==========================
    //! default constructor
    GroupWrapper();

    //---------------------------------------------------------------------
    //! copy constructor
    GroupWrapper(const GroupWrapper &o) = default;

    //---------------------------------------------------------------------
    //! move constructor
    GroupWrapper(GroupWrapper &&o) = default;


    //---------------------------------------------------------------------
    //! conversion copy constructor
    GroupWrapper(const hdf5::node::Group &group);

    operator hdf5::node::Group() const
    {
      return group_;
    }

    //---------------------------------------------------------------------
    //! conversion move constructor
    GroupWrapper(hdf5::node::Group &&group);

    //---------------------------------------------------------------------
    AttributeManagerWrapper attributes;

    //---------------------------------------------------------------------
    //!
    //! \brief create group
    //!
    //! Create a new group object. The method takes the name of the
    //! group (or the path relative to the actual group) and an optional
    //! argument with the Nexus class string of the new group. If the
    //! latter argument is not provided no NX_class attribute will be set
    //! on the newly created group.
    //!
    //! \throws nxgroup_error in case of problems
    //! \param n name of the new group
    //! \param nxclass optional argument with the Nexus class
    //! \return new instance of NXGroupWrapper
    //!
    hdf5::node::Group
    create_group(const std::string &n,
                 const std::string &nxclass=std::string()) const;

    //---------------------------------------------------------------------
    //!
    //! \brief create field
    //!
    //! This method creates a new NXField object in the file. It has two
    //! mandatory positional arguments: the name and the type_code of the
    //! new field. Further optional keyword arguments allow the creation
    //! of multidimensional fields and the creation of fields using
    //! filters for compressing data.
    //!
    //! A multidimensional field is created by setting the 'shape'
    //! argument to a tuple whose length defines the number of dimensions
    //! of the field and its elements the number of elements along each
    //! dimension.  The 'chunk' argument takes any sequence object an
    //! defines the shape of data chunks which are written contiguously
    //! to the file. Finally, the filter object can be used to pass a
    //! filter for data compression to the field.
    //!
    //! \throws nxfield_error in case of general problems with object
    //! creation
    //! \throws type_error if the type_code argument contains an invalid
    //! string
    //! \throws shape_mismatch_error if the rank of the chunk shape
    //! does not match the rank of the fields shape
    //!
    //! \param name name of the new field
    //! \param type_code numpy type code
    //! \param shape sequence object with the shape of the field
    //! \param chunk sequence object for the chunk shape
    //! \param filter a filter object for data compression
    //! \return instance of a field wrapper
    //!
    hdf5::node::Dataset
    create_field(const std::string &name,
                 const std::string &type_code,
                 const boost::python::object &shape=boost::python::object(),
                 const boost::python::object &chunk=boost::python::object(),
                 const boost::python::object &filter=boost::python::object()) const;

    //-------------------------------------------------------------------------
    //!
    //! \brief open an object
    //!
    //! This method opens an object and tries to figure out by itself
    //! what kind of object is has to deal with. Consequently it returns
    //! already a Python obeject of the appropriate type (which can be
    //! either NXGroup of NXField). If the method fails to determine the
    //! type of the return object an exception will be thrown.
    //!
    //! \throws nxgroup_error if opening the object fails
    //! \throws type_error if the return type cannot be determined
    //! \param n name of the object to open
    //! \return Python object for NXGroup or NXField.
    //!
    hdf5::node::Node open_by_name(const std::string &n) const;

    //---------------------------------------------------------------------
    //!
    //! \brief open object by index
    //!
    //! Opens a child of the group by its index.
    //! \throws index_error if the index exceeds the total number of child
    //! objects
    //! \throws nxgroup_error in case of errors while opening the object
    //! \param i index of the child
    //! \return child object
    //!
    hdf5::node::Node open_by_index(size_t i) const;

    //--------------------------------------------------------------------------
    //!
    //! \brief check for objects existance
    //!
    //! Returns true if the object defined by path 'n'.
    //! \return true if object exists, false otherwise
    //!
    bool exists(const std::string &n) const;

    //--------------------------------------------------------------------------
    //!
    //! \brief number of child objects
    //!
    //! Return the number of child objects linked below this group.
    //! \return number of child objects
    //!
    size_t nchildren() const;


    bool is_valid() const;


    void close();


    std::string filename() const;


    std::string name() const;


    hdf5::node::Node parent() const;


    size_t __len__() const;


    nexus::NodeIteratorWrapper __iter__() const;

    nexus::RecursiveNodeIteratorWrapper recursive();

    std::string path() const;

    void remove(const std::string &name) const;

};

//!
//! \brief convert nxgroup instances to their wrapper type
//!
//! Convert an instance of nxgroup to its corresponding wrapper type.
//!
//! \tparam GTYPE group type
//!

struct GroupToPythonObject
{
    //------------------------------------------------------------------------
    //!
    //! \brief conversion method
    //!
    //! \param v instance of bool_t
    //! \return Python boolean object
    //!
    static PyObject *convert(const hdf5::node::Group &group);
};



//!
//! \ingroup wrappers
//! \brief create NXGroup wrapper
//! 
//! Template function to create a new wrapper for an NXGroup type GType.
//! \param class_name name for the Python class
//!
void wrap_nxgroup(const char *class_name);

