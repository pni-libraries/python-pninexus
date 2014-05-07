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

#include <pni/io/nx/nxobject_type.hpp>
#include <pni/core/utilities.hpp>
#include <pni/io/nx/nxlink.hpp>

#include "NXWrapperHelpers.hpp"
#include "NXObjectMap.hpp"
#include "NXObjectWrapper.hpp"
#include "FieldCreator.hpp"
#include "ChildIterator.hpp"
#include "AttributeIterator.hpp"

/*! 
\ingroup wrappers  
\brief class tempalte for NXGroup wrapper

Class template to create wrappers for NXGroup types.
*/
template<typename GTYPE> class NXGroupWrapper:public NXObjectWrapper<GTYPE>
{
    public:
        //!wrapped type 
        typedef GTYPE type_t;
        //! wrapper for NXObject
        typedef NXObjectWrapper<type_t> object_wrapper_t;
        //! wrapper type for GTYPE
        typedef NXGroupWrapper<type_t> group_wrapper_t;
        //! field creator type
        typedef FieldCreator<type_t> field_creator_t;
        //================constructors and destructor==========================
        //! default constructor
        NXGroupWrapper():NXObjectWrapper<type_t>(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXGroupWrapper(const group_wrapper_t &o):
            NXObjectWrapper<type_t>(o)
        { }

        //----------------------------------------------------------------------
        //! move constructor
        NXGroupWrapper(group_wrapper_t &&o):
            NXObjectWrapper<type_t>(std::move(o))
        { }

        //----------------------------------------------------------------------
        //! conversion copy constructor
        explicit NXGroupWrapper(const type_t &g):NXObjectWrapper<type_t>(g){}

        //----------------------------------------------------------------------
        //! conversion move constructor
        explicit NXGroupWrapper(type_t &&g):NXObjectWrapper<type_t>(std::move(g))
        {}

        //----------------------------------------------------------------------
        //! destructor
        virtual ~NXGroupWrapper() { }

        //====================assignment operators==============================
        //!copy assignment 
        group_wrapper_t &operator=(const group_wrapper_t &o)
        {
            if(this != &o) object_wrapper_t::operator=(o);
            return *this;
        }

        //-----------------------------------------------------------------------
        //!move assignment
        group_wrapper_t &operator=(group_wrapper_t &&o)
        {
            if(this != &o) object_wrapper_t::operator=(std::move(o));
            return *this;
        }

        //----------------------------------------------------------------------
        /*! \brief create group 
        
        Create a new group object. The method takes the name of the group (or 
        the path relative to the actual group) and an optional argument with the 
        Nexus class string of the new group. If the latter argument is not 
        provided no NX_class attribute will be set on the newly created group.
        \throws NXGroupError in case of problems
        \param n name of the new group
        \param nxclass optional argument with the Nexus class
        \return new instance of NXGroupWrapper
        */
        NXGroupWrapper<typename NXObjectMap<type_t>::GroupType>
        create_group(const string &n,const string &nxclass=string()) const
        {
            typedef typename NXObjectMap<type_t>::GroupType l_group_t;
            typedef NXGroupWrapper<l_group_t> l_group_wrapper_t;

            l_group_wrapper_t group(this->_object.create_group(n,nxclass));
            return group;
        }

        //-------------------------------------------------------------------------
        /*! \brief create field

        This method creates a new NXField object in the file. It has two
        mandatory positional arguments: the name and the type_code of the new
        field. Further optional keyword arguments allow the creation of
        multidimensional fields and the creation of fields using filters for
        compressing data.
        A multidimensional field is created by setting the 'shape' argument 
        to a tuple whose length defines the number of dimensions of the field
        and its elements the number of elements along each dimension. 
        The 'chunk' argument takes any sequence object an defines the shape of
        data chunks which are written contiguously to the file. Finally, the
        filter object can be used to pass a filter for data compression to the
        field.
        \throws NXFieldError in case of general problems with object creation
        \throws TypeError if the type_code argument contains an invalid string
        \throws ShapeMissmatchError if the rank of the chunk shape does not
        match the rank of the fields shape
        \param name name of the new field
        \param type_code numpy type code 
        \param shape sequence object with the shape of the field
        \param chunk sequence object for the chunk shape
        \param filter a filter object for data compression
        \return instance of a field wrapper
        */
        typename field_creator_t::field_wrapper_t
         create_field(const string &name,const string &type_code,
                         const object &shape=list(),const object
                         &chunk=list(),const object &filter=object()) const
        {
            typedef typename field_creator_t::field_wrapper_t field_wrapper_t;
            field_creator_t creator(name,List2Container<shape_t>(list(shape)),
                                    List2Container<shape_t>(list(chunk)),filter);

            field_wrapper_t wrapper = creator.create(this->_object,type_code);
            return wrapper;
        }

        //-------------------------------------------------------------------------
        /*! \brief open an object

        This method opens an object and tries to figure out by itself what kind
        of object is has to deal with. Consequently it returns already a Python
        obeject of the appropriate type (which can be either NXGroup of
        NXField). If the method fails to determine the type of the return object
        an exception will be thrown.
        \throws NXGroupError if opening the object fails
        \throws TypeError if the return type cannot be determined
        \param n name of the object to open
        \return Python object for NXGroup or NXField.
        */
        object open_by_name(const string &n) const
        {
            typedef typename NXObjectMap<type_t>::ObjectType object_t;
            typedef typename NXObjectMap<type_t>::GroupType l_group_t;
            typedef NXGroupWrapper<l_group_t> l_group_wrapper_t;
            typedef typename field_creator_t::field_t field_t;
            typedef typename field_creator_t::field_wrapper_t field_wrapper_t;
           
            //open the NXObject 
            object_t nxobject = this->_object.open(n);

            //we use here copy construction thus we do not have to care
            //of the original nxobject goes out of scope and gets destroyed.
            if(nxobject.object_type() == pni::io::nx::nxobject_type::NXFIELD)
               return object(field_wrapper_t(field_t(nxobject)));

            if(nxobject.object_type() == pni::io::nx::nxobject_type::NXGROUP)
                return object(l_group_wrapper_t(l_group_t(nxobject)));

            nxobject.close();
            //this here is to avoid compiler warnings
            return object();

        }

        //---------------------------------------------------------------------
        /*! \brief open object by index

        Opens a child of the group by its index.
        \throws IndexError if the index exceeds the total number of child
        objects
        \throws NXGroupError in case of errors while opening the object
        \param i index of the child
        \return child object
        */
        object open(size_t i) const
        {
            typedef typename NXObjectMap<type_t>::ObjectType object_t;
            object_t nxobject = this->_object.open(i);

            return open_by_name(nxobject.path());
        }

        //--------------------------------------------------------------------------
        /*! \brief check for objects existance
        
        Returns true if the object defined by path 'n'. 
        \return true if object exists, false otherwise
        */
        bool exists(const string &n) const { return this->_object.exists(n); }

        //--------------------------------------------------------------------------
        /*! \brief number of child objects

        Return the number of child objects linked below this group.
        \return number of child objects
        */
        size_t nchildren() const { return this->_object.nchildren(); }


        //---------------------------------------------------------------------
        /*! \brief create links

        Exposes only one of the three link creation methods from the original
        NXGroup object.
        */
        void link(const string &p,const string &n) const
        {
            pni::io::nx::link(p,this->_object,n);

            //this->_object.link(p,n);
        }

        //----------------------------------------------------------------------
        /*! \brief get child iterator

        Returns an iterator over all child objects linked below this group.
        \return instance of ChildIterator
        */
        ChildIterator<group_wrapper_t,object> get_child_iterator() const
        {
            return ChildIterator<group_wrapper_t,object>(*this);
        }

};

static const char __group_open_docstr[] = 
"Opens an existing object. The object is identified by its name (path)."
"The method tries to figure out which type of object to return by itself.\n\n"
"Required arguments:\n"
"\tn............name (path) to the object to open\t\t"
"Return value:\n"
"\tinstance of a new object.";

static const char __group_create_group_docstr[] = 
"Create a new group. The location where the group is created can either be\n"
"relative to this group (if the name does not start with a /) or absolute\n"
"(if the groups name starts with a /). In addition the optional keyword-\n"
"argument 'nxclass' allows setting the NX_class attribute directly during\n"
"group creation. If 'nxclass' is not set the attribute is not written.\n\n"
"Required arguments:\n"
"\tname ............. name (path) of the new group (relative or absolute)\n\n"
"Optional keyword arguments:\n"
"\tnxclass .......... sets the NX_class attribute of the group\n\n"
"Return value:\n"
"\tnew instance of a group object";

static const char __group_create_field_docstr[] = 
"Create a new field. The location of the new field can be relative to this\n"
"group (if name does not start with /) or absolute (name must start with /).\n"
"If only the two mandatory positional arguments are passed a simple scalar\n"
"field will be created. To create a multidimensional field 'shape' muste be\n"
"a sequence object defining the number of elements along each dimension.\n"
"The 'chunk' argument takes a sequence object which  determines the shape \n"
"of the data chunks being writen to disk. Finally, the 'filter' takes a \n"
"filter object for the compression of data stored in the field.\n\n"
"Required arguments:\n"
"\tname .................. name (path) of the new field\n"
"\ttype_code ............. numpy type code for the field\n\n"
"Optional keyword arguments:\n"
"\tshape ................. sequence object defining the shape of a multi-\n"
"\t                        dimensional field\n"
"\tchunk ................. sequence object defining the chunk shape\n"
"\tfilter ................ filter object for data compression\n\n"
"Return value:\n"
"\tnew instance of a field object";

static const char __group_exists_docstr[] = 
"Checks if the object determined by 'name' exists. 'name' can be either\n"
"an absolute or relative path. If the object identified by 'name' exists\n"
"the method returns true, false otherwise.\n\n"
"Required arguments:\n"
"\tname .................. name (path) of the object to check\n\n"
"Return value:\n"
"\t true if object exists, false otherwise";
;
static const char __group_link_docstr[] = 
"Create an internal or external link. The first argument is the path to the \n"
"object to which a new link shall be created. The second argument is a path \n"
"relative to this group or absolute describing the name of the new link. An \n"
"external link can be created by prefixing 'p' with the name of the file \n"
"where to find the object: filename:/path/to/object."
"\n\n"
"Required arguments:\n"
"\tp .................... object to link to\n"
"\tn .................... name of the new link\n" ;
static const char __group_nchilds_docstr[] = 
"Read only property providing the number of childs linked below this group.";
static const char __group_childs_docstr[] = 
"Read only property providing a sequence object which can be used to iterate\n"
"over all childs linked below this group.";

/*!
\ingroup wrappers
\brief create NXGroup wrapper

Template function to create a new wrapper for an NXGroup type GType.
\param class_name name for the Python class
*/
template<typename GTYPE> void wrap_nxgroup(const string &class_name)
{
    typedef typename NXGroupWrapper<GTYPE>::group_wrapper_t group_wrapper_t;
    typedef typename NXObjectWrapper<GTYPE>::object_wrapper_t object_wrapper_t;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    class_<group_wrapper_t,bases<object_wrapper_t> >(class_name.c_str())
        .def(init<>())
        .def("open",&group_wrapper_t::open_by_name,__group_open_docstr)
        .def("__getitem__",&group_wrapper_t::open)
        .def("__getitem__",&group_wrapper_t::open_by_name)
        .def("create_group",&group_wrapper_t::create_group,
                ("n",arg("nxclass")=string()),__group_create_group_docstr)
        .def("create_field",&group_wrapper_t::create_field,
                ("name","type_code",arg("shape")=list(),arg("chunk")=list(),
                 arg("filter")=object()),__group_create_field_docstr)
        .def("exists",&group_wrapper_t::exists,__group_exists_docstr)
        .def("link",&group_wrapper_t::link,__group_link_docstr)
        .def("__iter__",&group_wrapper_t::get_child_iterator,__group_childs_docstr)
        .add_property("nchildren",&group_wrapper_t::nchildren,__group_nchilds_docstr)   
        .add_property("children",&group_wrapper_t::get_child_iterator,__group_childs_docstr)
        ;
#pragma GCC diagnostic pop
}

