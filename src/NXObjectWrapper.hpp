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
#include<Python.h>
}

#include <boost/python.hpp>
#include <pni/core/types.hpp>

#include "NXObjectMap.hpp"
#include "NXAttributeWrapper.hpp"
#include "AttributeIterator.hpp"
#include "AttributeCreator.hpp"

using namespace pni::core;
using namespace boost::python;


/*! 
\ingroup wrappers  
\brief template wrapps NXObject classes

This class-template provides a wrapper for NXObject types. It collates all
methods common to all basic Nexus classes. Most of the methods are just
delegates. They should return only types that can be converted into the
appropriate Python type. A type that should be wrapped by this class has to
satisfy several requierements
\li it must be copyable
\li and it must be moveable

*/
template<typename OTYPE> class NXObjectWrapper
{   
    protected:
        //object is not defined private here. The intention of this class is 
        //not encapsulation but rather reducing the typing effort for the 
        //child classes by collecting here all common methods. 
        OTYPE _object; //!< original object that shall be wrapped
    public:
        //=================public types========================================
        //! wrapped type
        typedef OTYPE type_t;
        //! object wrapper type
        typedef NXObjectWrapper<type_t> object_wrapper_t;
        //! attribute type
        typedef NXAttributeWrapper<typename NXObjectMap<type_t>::AttributeType>
            attribute_type; 
        //================constructors and destructor==========================
        //! default constructor
        NXObjectWrapper():_object(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXObjectWrapper(const object_wrapper_t &o):_object(o._object) {}

        //---------------------------------------------------------------------
        //! move constructor
        NXObjectWrapper(object_wrapper_t &&o):_object(std::move(o._object)) {}

        //---------------------------------------------------------------------
        /*! 
        \brief construct from type_t

        Construct the object from an instance of type_t (the wrapped type). 
        In this case the original instance of type_t will be copied. 
        \param o reference to instance of type_t
        */
        explicit NXObjectWrapper(const type_t &o):_object(o){}

        //---------------------------------------------------------------------
        //! move conversion constructor from wrapped object
        explicit NXObjectWrapper(type_t &&o):_object(std::move(o)){ }

        //---------------------------------------------------------------------
        //! destructor
        virtual ~NXObjectWrapper()
        {
            //there is nothing we have to do here. As the wrapped objects are
            //all first class objects they will be destroyed automatically when
            //their wrapper object gets destroyed.
        }

        //==================assignment operators===============================
        //!move assignment
        object_wrapper_t &operator=(object_wrapper_t &&o)
        {
            if(this != &o) this->_object = std::move(o._object);
            return *this;
        }

        //---------------------------------------------------------------------
        //!copy assignment
        object_wrapper_t &operator=(const object_wrapper_t &o)
        {
            if(this != &o) this->_object = o._object;
            return *this;
        }


        //======================object methods=================================
        //! obtain base name
        string base() const { return _object.base(); }

        //----------------------------------------------------------------------
        //! obtain name
        string name() const { return _object.name(); }

        //----------------------------------------------------------------------
        //! obtain path
        string path() const { return _object.path(); }

        //----------------------------------------------------------------------
        //! get validity status
        bool is_valid() const { return _object.is_valid(); }

        //----------------------------------------------------------------------
        //! close the object
        void close() { this->_object.close(); }

        //---------------------------------------------------------------------
        /*! \brief create attribute

        Create an attribute attached to this object. By default attributes are
        overwritten if they already exist.
        \throws TypeError if the type_code cannot be handled
        \throws NXAttributeError in case of other attribute errors
        \param name name of the attribute
        \param type_code numpy type code for the attribute
        \param shape list or tuple with shape information.
        */
        attribute_type create_attribute(const string &name,const string
                &type_code,const object &shape=list())
            const
        {
            //first we need to decide wether we need a scalar or an array 
            //attribute
            AttributeCreator<attribute_type>
                creator(name,List2Container<std::vector<size_t> >(list(shape)));

            return creator.create(this->_object,type_code);
        }

        //---------------------------------------------------------------------
        /*! \brief opens attribute by name

        \param n name of the attribute
        \return attribute instance
        */
        attribute_type open_attr(const string &n) const
        {
            return attribute_type(this->_object.attr(n));
        }

        //---------------------------------------------------------------------
        //! return number of attributes
        size_t nattrs() const { return this->_object.nattr(); }

        //---------------------------------------------------------------------
        /*! \brief return attribute by index

        Returns the attribute by its index rather than by its name.
        \param i attribute index
        \return attribute instance
        */
        attribute_type open_attr_by_id(size_t i) const
        {
            return attribute_type(this->_object.attr(i));
        }

        //---------------------------------------------------------------------
        /*! \brief return attribute iterator

        Method returns an iterator over the attributes attached to this object.
        \return attribute iterator
        */
        AttributeIterator<object_wrapper_t,attribute_type> 
            get_attribute_iterator() const
        {
            return
                AttributeIterator<object_wrapper_t,attribute_type>(*this);
        }
};

//----------------------docstrings for the wrapped objects----------------------
static const char __object_name_docstr[] =
"Read only property providing the name of the object as a string. As all\n"
"objects can be identified by a UNIX like path this is the very last part\n"
"of the path (on a file-system this would be the file or the directory name\n"
"the path is pointing to.";

static const char __object_path_docstr[] = 
"Read only property providing the full path of the object within the\n"
"file.";

static const char __object_base_docstr[] = 
"Read only property providing the base name (the path without its very \n"
"last element) of the object.";

static const char __object_valid_docstr[] = 
"Read only property of a boolean value. If true the object is valid and \n"
"can safely be used. Otherwise (when wrong) something went wrong and \n"
"each operation on the object will fail.";

static const char __object_nattrs_docstr[] =
"Read only property with the number of attributes attached to this object.";

static const char __object_attr_create_docstr[] = 
"Creates a new attribute on this object. An attribute can either be a scalar\n"
"or a multidimensional array. The latter can be created if the \n"
"keyword-argument shape is used.\n\n"
"Required positional arguments:\n"
"\tnamei...............the name of the attribute\n"
"\ttype_code...........the numpy type-code of the attribute\n\n"
"Optional keyword arguments:\n"
"\tshape\t\ta list or tuple with the shape of the attribute\n\n"
"Return value:\n"
"\tattribute instance";

static const char __object_attr_open_docstr[] = 
"Open an existing attribute.\n\n"
"Required positional arguments:\n"
"\tname................name of the attribute\n\n"
"Return value:\n"
"\tattribute instance.";

static const char __object_close_docstr[] = "Closes an open object.";
static const char __object_attributes_docstr[] = 
"Read only property providing an sequence object to iterate over all\n"
"attributes attached to this object.";


/*! 
\ingroup wrappers  
\brief tempalte function for NXObject wrappers

Template function to create NXObject wrappers. The template parameter determines
the type to be wrapped.
\param class_name name of the new Python class
*/
template<typename OTYPE> void wrap_nxobject(const string &class_name)
{
    typedef typename NXObjectWrapper<OTYPE>::object_wrapper_t object_wrapper_t;
    typedef class_<object_wrapper_t> wrapper_class;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    wrapper_class(class_name.c_str())
        .def(init<>())
        .add_property("name", &object_wrapper_t::name,__object_name_docstr)
        .add_property("path", &object_wrapper_t::path,__object_path_docstr)
        .add_property("base", &object_wrapper_t::base,__object_base_docstr)
        .add_property("valid", &object_wrapper_t::is_valid,__object_valid_docstr)
        .add_property("nattrs", &object_wrapper_t::nattrs,__object_nattrs_docstr)
        .def("attr", &object_wrapper_t::create_attribute,("name","type_code", arg("shape")=list()),__object_attr_create_docstr)
        .def("attr", &object_wrapper_t::open_attr,__object_attr_open_docstr)
        .def("close", &object_wrapper_t::close,__object_close_docstr)
        .add_property("attributes",&object_wrapper_t::get_attribute_iterator,__object_attributes_docstr)
        ;
#pragma GCC diagnostic pop
}


