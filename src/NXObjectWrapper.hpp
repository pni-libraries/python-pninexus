/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of libpninx-python.
 *
 * libpninx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * libpninx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libpninx.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Definition of the NXObjectWrapper template.
 *
 * Created on: Feb 17, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __NXOBJECTWRAPPER_HPP__
#define __NXOBJECTWRAPPER_HPP__

extern "C"{
#include<Python.h>
}

#include <boost/python.hpp>
#include <pni/utils/Types.hpp>

#include "NXObjectMap.hpp"
#include "NXAttributeWrapper.hpp"
#include "AttributeIterator.hpp"
#include "AttributeCreator.hpp"

using namespace pni::utils;
using namespace boost::python;


/*! 
\ingroup wrappers  
\brief template wrapps NXObject classes

This class-template provides a wrapper for NXObject types.
*/
template<typename OType> class NXObjectWrapper
{   
    protected:
        //object is not defined private here. The intention of this class is 
        //not encapsulation but rather reducing the writing effort for the 
        //child classes by collecting here all common methods. 
        OType _object; //!< original object that shall be wrapped
    public:
        typedef NXAttributeWrapper<typename NXObjectMap<OType>::AttributeType>
            attribute_type; //!< type for attributes
        //================constructors and destructor==========================
        //! default constructor
        NXObjectWrapper():_object(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXObjectWrapper(const NXObjectWrapper<OType> &o):
            _object(o._object)
        { }

        //---------------------------------------------------------------------
        //! move constructor
        NXObjectWrapper(NXObjectWrapper<OType> &&o):
            _object(std::move(o._object)) 
        { }

        //---------------------------------------------------------------------
        //! copy conversion constructor from wrapped object
        explicit NXObjectWrapper(const OType &o):_object(o){}

        //---------------------------------------------------------------------
        //! move conversion constructor from wrapped object
        explicit NXObjectWrapper(OType &&o):_object(std::move(o)){ }

        //---------------------------------------------------------------------
        //! destructor
        virtual ~NXObjectWrapper()
        {
            std::cout<<"Destory object wrapper!"<<std::endl;
            //close the object on wrapper destruction
            this->_object.close();
        }

        //==================assignment operators===============================
        //!move assignment
        NXObjectWrapper<OType> &operator=(NXObjectWrapper<OType> &&o)
        {
            if(this != &o) _object = std::move(o._object);
            return *this;
        }

        //---------------------------------------------------------------------
        //!copy assignment
        NXObjectWrapper<OType> &operator=(const NXObjectWrapper<OType> &o)
        {
            if(this != &o) _object = o._object;
            return *this;
        }


        //======================object methods=================================
        //! obtain base name
        String base() const { return _object.base(); }

        //----------------------------------------------------------------------
        //! obtain name
        String name() const { return _object.name(); }

        //----------------------------------------------------------------------
        //! obtain path
        String path() const { return _object.path(); }

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
        attribute_type create_attribute(const String &name,const String
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
        attribute_type open_attr(const String &n) const
        {
            return attribute_type(this->_object.attr(n));
        }

        //---------------------------------------------------------------------
        //! return number of attributes
        size_t nattrs() const
        {
            return this->_object.nattr();
        }

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
        AttributeIterator<NXObjectWrapper<OType>,attribute_type> 
            get_attribute_iterator() const
        {
            return
                AttributeIterator<NXObjectWrapper<OType>,attribute_type>(*this);
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
template<typename OType> void wrap_nxobject(const String &class_name)
{
    typedef NXObjectWrapper<OType> wrapper;
    typedef class_<NXObjectWrapper<OType> > wrapper_class;

    wrapper_class(class_name.c_str())
        .def(init<>())
        .add_property("name", &wrapper::name,__object_name_docstr)
        .add_property("path", &wrapper::path,__object_path_docstr)
        .add_property("base", &wrapper::base,__object_base_docstr)
        .add_property("valid", &wrapper::is_valid,__object_valid_docstr)
        .add_property("nattrs", &wrapper::nattrs,__object_nattrs_docstr)
        .def("attr", &wrapper::create_attribute,("name","type_code", arg("shape")=list()),__object_attr_create_docstr)
        .def("attr", &wrapper::open_attr,__object_attr_open_docstr)
        .def("close", &wrapper::close,__object_close_docstr)
        .add_property("attributes",&wrapper::get_attribute_iterator,__object_attributes_docstr)
        ;
}



#endif
