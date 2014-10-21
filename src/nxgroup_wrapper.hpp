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

#include <pni/io/nx/nx.hpp>
#include <pni/core/utilities.hpp>
#include <pni/io/nx/nxobject_traits.hpp>
#include <pni/io/nx/algorithms/create_field.hpp>

#include "nxwrapper_utils.hpp"
#include "nxfield_wrapper.hpp"
#include "child_iterator.hpp"

using namespace pni::io::nx;

//! 
//! \ingroup wrappers  
//! \brief class tempalte for NXGroup wrapper
//! 
//! Class template to create wrappers for NXGroup types.
//!
template<typename GTYPE> 
class nxgroup_wrapper
{
    public:
        typedef GTYPE group_type;

        static const nximp_code imp_id = nximp_code_map<group_type>::icode;
        typedef typename nxobject_trait<imp_id>::field_type field_type;
        typedef typename nxobject_trait<imp_id>::object_type object_type;
        typedef typename nxobject_trait<imp_id>::deflate_type deflate_type;
        typedef nxfield_wrapper<field_type> field_wrapper_type;
        typedef nxgroup_wrapper<group_type> wrapper_type;
    private:
        group_type _group;
    public:
        //================constructors and destructor==========================
        //! default constructor
        nxgroup_wrapper(){}

        //---------------------------------------------------------------------
        //! copy constructor
        nxgroup_wrapper(const wrapper_type &o): _group(o._group)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        nxgroup_wrapper(wrapper_type &&o):_group(std::move(o._group))
        {}

        //---------------------------------------------------------------------
        //! conversion copy constructor
        explicit nxgroup_wrapper(const group_type &g):_group(g)
        {}

        //---------------------------------------------------------------------
        //! conversion move constructor
        explicit nxgroup_wrapper(group_type &&g):_group(std::move(g))
        {}

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
        group_type
        create_group(const string &n,const string &nxclass=string()) const
        {
            return group_type(_group.create_group(n,nxclass));
        }

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
        field_wrapper_type
        create_field(const string &name,
                     const string &type_code,
                     const object &shape=object(),
                     const object &chunk=object(),
                     const object &filter=object()) const
        {
            using pni::io::nx::create_field;
            
            type_id_t type_id;
            try
            {
                type_id = type_id_from_str(type_code);
            }
            catch(key_error &error)
            {
                //forward exception
                error.append(EXCEPTION_RECORD); throw error;
            }

            if(filter.is_none())
            {
                //create a field without a filter
                if(shape.is_none())
                {
                    //this corresponds to create_field<T>(name);
                    field_type field = create_field(object_type(_group),
                                                    type_id,name);
                    return field_wrapper_type(field);
                }
                else if(!shape.is_none() && chunk.is_none())
                {
                    auto s = List2Container<shape_t>(list(shape));
                    field_type field = create_field(object_type(_group),
                                                     type_id,name,s);
                    return field_wrapper_type(field);
                }
                else if(!shape.is_none() && !chunk.is_none())
                {
                    auto s = List2Container<shape_t>(list(shape));
                    auto c = List2Container<shape_t>(list(chunk));
                    field_type field = create_field(object_type(_group),
                                                    type_id,name,s,c);
                    return field_wrapper_type(field);
                }
                else
                {
                    throw pni::io::object_error(EXCEPTION_RECORD,
                            "Cannot create field from arguments!");
                }
            }
            else
            {
                //extract filter
                extract<deflate_type> deflate_object(filter);

                if(!deflate_object.check())
                    throw type_error(EXCEPTION_RECORD,
                                     "Filter is not an instance of filter class!");

                if(!shape.is_none() && chunk.is_none())
                {
                    auto s = List2Container<shape_t>(list(shape));
                    shape_t c(s);
                    c.front() = 0;
                    field_type field = create_field(object_type(_group),
                                                    type_id,
                                                    name,s,c,
                                                    deflate_object());
                    return field_wrapper_type(field);
                }
                else if(!shape.is_none() && !chunk.is_none())
                {
                    auto s = List2Container<shape_t>(list(shape));
                    auto c = List2Container<shape_t>(list(chunk));
                    field_type field = create_field(object_type(_group),
                                                    type_id,name,s,c,
                                                    deflate_object());
                    return field_wrapper_type(field);
                }
                else
                {
                    throw pni::io::object_error(EXCEPTION_RECORD,
                            "Cannot create field from arguments!");
                }
                    //throw an exception here
            }

        }

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
        object open_by_name(const string &n) const
        {
           
            //open the NXObject 
            object_type nxobject = _group.at(n);

            //we use here copy construction thus we do not have to care
            //of the original nxobject goes out of scope and gets destroyed.
            if(is_field(nxobject)) 
                return object(field_wrapper_type(as_field(nxobject)));

            if(is_group(nxobject)) 
                return object(wrapper_type(as_group(nxobject)));

            //this here is to avoid compiler warnings
            return object();

        }

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
        object open(size_t i) const
        {
            object_type nxobject = _group.at(i);

            if(is_field(nxobject)) 
                return object(field_wrapper_type(as_field(nxobject)));

            if(is_group(nxobject)) 
                return object(wrapper_type(as_group(nxobject)));

            return object();
        }

        //--------------------------------------------------------------------------
        //! 
        //! \brief check for objects existance
        //! 
        //! Returns true if the object defined by path 'n'. 
        //! \return true if object exists, false otherwise
        //!
        bool exists(const string &n) const { return _group.has_child(n); }

        //--------------------------------------------------------------------------
        //!
        //! \brief number of child objects
        //!
        //! Return the number of child objects linked below this group.
        //! \return number of child objects
        //!
        size_t nchildren() const { return _group.size(); }

        //--------------------------------------------------------------------
        bool is_valid() const { return _group.is_valid(); } 

        //--------------------------------------------------------------------
        void close() { _group.close(); }

        //--------------------------------------------------------------------
        string filename() const { return _group.filename(); }

        //--------------------------------------------------------------------
        string name() const { return _group.name(); }

        //--------------------------------------------------------------------
        wrapper_type parent() const
        {
            return wrapper_type(_group.parent());
        }

        //--------------------------------------------------------------------
        size_t __len__() const { return _group.size(); }

        //---------------------------------------------------------------------
        //!
        //! \brief create links
        //!
        //! Exposes only one of the three link creation methods from the 
        //! original NXGroup object.
        //!
        void link(const string &p,const string &n) const
        {
            pni::io::nx::link(p,_group,n);
        }

        //----------------------------------------------------------------------
        //!
        //! \brief get child iterator
        //!
        //! Returns an iterator over all child objects linked below this 
        //! group.
        //!
        //! \return instance of ChildIterator
        //!
        child_iterator<wrapper_type> get_child_iterator() const
        {
            return child_iterator<wrapper_type>(*this);
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

//!
//! \ingroup wrappers
//! \brief create NXGroup wrapper
//! 
//! Template function to create a new wrapper for an NXGroup type GType.
//! \param class_name name for the Python class
//!
template<typename GTYPE> void wrap_nxgroup()
{
    typedef nxgroup_wrapper<GTYPE> wrapper_type;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    class_<wrapper_type>("nxgroup")
        .def(init<>())
        .def("open",&wrapper_type::open_by_name,__group_open_docstr)
        .def("__getitem__",&wrapper_type::open)
        .def("__getitem__",&wrapper_type::open_by_name)
        .def("create_group",&wrapper_type::create_group,
                ("n",arg("nxclass")=string()),__group_create_group_docstr)
        .def("create_field",&wrapper_type::create_field,
                ("name","type_code",arg("shape")=object(),arg("chunk")=object(),
                 arg("filter")=object()),__group_create_field_docstr)
        .def("exists",&wrapper_type::exists,__group_exists_docstr)
        .def("close",&wrapper_type::close)
        .def("is_valid",&wrapper_type::is_valid)
        .def("__len__",&wrapper_type::__len__)

        .def("link",&wrapper_type::link,__group_link_docstr)
        .def("__iter__",&wrapper_type::get_child_iterator,__group_childs_docstr)
        .add_property("nchildren",&wrapper_type::nchildren,__group_nchilds_docstr)   
        .add_property("children",&wrapper_type::get_child_iterator,__group_childs_docstr)
        .add_property("filename",&wrapper_type::filename)
        .add_property("name",&wrapper_type::name)
        .add_property("parent",&wrapper_type::parent)
        ;
#pragma GCC diagnostic pop
}

