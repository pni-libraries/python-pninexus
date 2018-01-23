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
#include <pni/io/nx/nx.hpp>
#include <pni/core/utilities.hpp>
#include <pni/io/nx/nxobject_traits.hpp>
#include <pni/io/nx/algorithms/create_field.hpp>
#include <pni/io/nx/algorithms/get_path.hpp>
#include <pni/io/nx/flat_group.hpp>

#include <core/utils.hpp>
#include "child_iterator.hpp"
#include "nxattribute_manager_wrapper.hpp"
#include "utils.hpp"
#include "rec_group_iterator.hpp"

namespace nexus {
//! 
//! \ingroup wrappers  
//! \brief class tempalte for NXGroup wrapper
//! 
//! Class template to create wrappers for NXGroup types.
//!
class GroupWrapper
{
    public:
        typedef GTYPE group_type;
        typedef nxgroup_wrapper<group_type> wrapper_type;

        static const pni::io::nx::nximp_code imp_id = 
                     pni::io::nx::nximp_code_map<group_type>::icode;
        typedef typename pni::io::nx::nxobject_trait<imp_id>::field_type field_type;
        typedef typename pni::io::nx::nxobject_trait<imp_id>::object_type object_type;
        typedef typename pni::io::nx::nxobject_trait<imp_id>::deflate_type deflate_type;

        typedef decltype(group_type::attributes) attribute_manager_type;
        typedef nxattribute_manager_wrapper<attribute_manager_type>
            attribute_manager_wrapper;

    private:
        group_type _group;
        size_t _index;
    public:
        //================constructors and destructor==========================
        //! default constructor
        nxgroup_wrapper():
            _group(),
            _index(0),
            attributes(_group.attributes)
        {}

        //---------------------------------------------------------------------
        //! copy constructor
        nxgroup_wrapper(const wrapper_type &o): 
            _group(o._group),
            _index(o._index),
            attributes(_group.attributes)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        nxgroup_wrapper(wrapper_type &&o):
            _group(std::move(o._group)),
            _index(o._index),
            attributes(_group.attributes)
        {}

        //---------------------------------------------------------------------
        //! conversion copy constructor
        explicit nxgroup_wrapper(const group_type &g):
            _group(g),
            _index(0),
            attributes(_group.attributes)
        {}

        operator group_type() const
        {
            return _group;
        }

        //---------------------------------------------------------------------
        //! conversion move constructor
        explicit nxgroup_wrapper(group_type &&g):
            _group(std::move(g)),
            _index(0),
            attributes(_group.attributes)
        {}

        //---------------------------------------------------------------------
        attribute_manager_wrapper attributes;

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
        create_group(const pni::core::string &n,
                     const pni::core::string &nxclass=pni::core::string()) const
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
        field_type
        create_field(const pni::core::string &name,
                     const pni::core::string &type_code,
                     const boost::python::object &shape=boost::python::object(),
                     const boost::python::object &chunk=boost::python::object(),
                     const boost::python::object &filter=boost::python::object()) const
        {
            using namespace boost::python;
            using pni::io::nx::create_field;
            
            pni::core::type_id_t type_id;
            try
            {
                type_id = pni::core::type_id_from_str(type_code);
            }
            catch(pni::core::key_error &error)
            {
                //forward exception
                error.append(EXCEPTION_RECORD); 
                throw error;
            }

            field_type field;
            object_type parent(_group);
            shapes_type shapes = get_shapes(shape,chunk);
            if(filter.is_none())
                field = create_field(parent, type_id,
                                     name,
                                     shapes.first,
                                     shapes.second);
            else
            {
                //extract filter
                extract<deflate_type> deflate_object(filter);

                if(!deflate_object.check())
                    throw pni::core::type_error(EXCEPTION_RECORD,
                                     "Filter is not an instance of filter class!");
            
                field = create_field(parent,type_id,name,shapes.first,
                                     shapes.second,deflate_object());
            }
            return field;

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
        object_type open_by_name(const pni::core::string &n) const
        {
            return _group.at(n);
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
        object_type open_by_index(size_t i) const
        {
            return _group.at(i);
        }

        //--------------------------------------------------------------------------
        //! 
        //! \brief check for objects existance
        //! 
        //! Returns true if the object defined by path 'n'. 
        //! \return true if object exists, false otherwise
        //!
        bool exists(const pni::core::string &n) const 
        { 
            return _group.has_child(n); 
        }

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
        pni::core::string filename() const { return _group.filename(); }

        //--------------------------------------------------------------------
        pni::core::string name() const { return _group.name(); }

        //--------------------------------------------------------------------
        object_type parent() const
        {
            return _group.parent();
        }

        //--------------------------------------------------------------------
        size_t __len__() const { return _group.size(); }

        //----------------------------------------------------------------------
        void increment()
        {
            _index ++;
        }

        //----------------------------------------------------------------------
        boost::python::object __iter__() const
        {
            return boost::python::object(this);
        }

        //----------------------------------------------------------------------
        object_type next()
        {
            if(_index >= _group.size())
            {
                throw(ChildIteratorStop());
                return object_type();
            }

            object_type child(_group[_index]);
            increment();

            return child;
        }

        //--------------------------------------------------------------------
        rec_group_iterator<GTYPE> recursive() 
        {
            typedef rec_group_iterator<GTYPE> iterator_type; 
            typedef typename iterator_type::group_type group_type;
            typedef typename iterator_type::group_ptr ptr_type; 
       
            //can use here a shared pointer which we pass around
            ptr_type ptr(new group_type(make_flat(_group)));
            return rec_group_iterator<GTYPE>(ptr,0);
        }

        //--------------------------------------------------------------------
        pni::core::string path() const
        {
            return pni::io::nx::get_path(_group);
        }

        void remove(const pni::core::string &name) const
        {
            _group.remove(name);
        }

};

} // namespace nexus

static const char __group_open_docstr[] = 
"Opens an existing object \n"
"\n"
"The object is identified by its name (path).\n"
"\n"
":param str n: name (path) to the object to open\n"
":return: the requested child object \n"
":rtype: either an instance of :py:class:`nxfield`, :py:class:`nxgroup`, or "
":py:class:`nxlink`\n"
":raises KeyError: if the child could not be found\n"
;

static const char __group_exists_docstr[] = 
"Checks if the object determined by 'name' exists\n"
"\n"
"'name' can be either an absolute or relative path. If the object "
"identified by 'name' exists the method returns true, false otherwise.\n"
"\n"
":param str name: name of the object to check\n"
":return: :py:const:`True` if the object exists, :py:const:`False` otherwise\n"
":rtype: bool"
;

static const char __group_nchilds_docstr[] = 
"Read only property providing the number of childs linked below this group.";
static const char __group_childs_docstr[] = 
"Read only property providing a sequence object which can be used to iterate\n"
"over all childs linked below this group.";

static const pni::core::string nxgroup_filename_doc = 
"Read only property returning the name of the file the group belongs to\n";

static const pni::core::string nxgroup_name_doc = 
"Read only property returning the name of the group\n";

static const pni::core::string nxgroup_parent_doc = 
"Read only property returning the parent group of this group\n";

static const pni::core::string nxgroup_size_doc = 
"Read only property returing the number of links below this group\n";

static const pni::core::string nxgroup_attributes_doc = 
"Read only property with the attribute manager for this group\n";

static const pni::core::string nxgroup_recursive_doc = 
"Read only property returning a recursive iterator for this group\n";

static const pni::core::string nxgroup_path_doc = 
"Read only property returning the NeXus path for this group\n";

static const pni::core::string nxgroup_close_doc = 
"Close this group";

static const pni::core::string nxgroup_is_valid_doc = 
"Read only property returning :py:const:`True` if this instance is a valid"
" NeXus object";

static const pni::core::string nxgroup__get_item_by_index_doc = 
"Get child by index";

static const pni::core::string nxgroup__get_item_by_name_doc = 
"Get child by name";

//!
//! \ingroup wrappers
//! \brief create NXGroup wrapper
//! 
//! Template function to create a new wrapper for an NXGroup type GType.
//! \param class_name name for the Python class
//!
template<typename GTYPE> void wrap_nxgroup()
{
    using namespace boost::python;

    typedef nxgroup_wrapper<GTYPE> wrapper_type;
    typedef rec_group_iterator<GTYPE> rec_group_iterator_type;

    class_<rec_group_iterator_type>("nxgroup_rec_iterator")
        .def("increment",&rec_group_iterator_type::increment)
        .def("__iter__",&rec_group_iterator_type::__iter__)
#if PY_MAJOR_VERSION >= 3
        .def("__next__",&rec_group_iterator_type::next);
#else
        .def("next",&rec_group_iterator_type::next);
#endif

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif
    class_<wrapper_type>("nxgroup")
        .def(init<>())
        .def("open",&wrapper_type::open_by_name,__group_open_docstr)
        .def("__getitem__",&wrapper_type::open_by_index,nxgroup__get_item_by_index_doc.c_str())
        .def("__getitem__",&wrapper_type::open_by_name)
        .def("_create_group",&wrapper_type::create_group,
                ("n",arg("nxclass")=pni::core::string()))
        .def("__create_field",&wrapper_type::create_field,
                ("name","type",arg("shape")=object(),arg("chunk")=object(),
                 arg("filter")=object()))
        .def("exists",&wrapper_type::exists,__group_exists_docstr)
        .def("close",&wrapper_type::close,nxgroup_close_doc.c_str())
        .add_property("is_valid",&wrapper_type::is_valid,nxgroup_is_valid_doc.c_str())
        .def("__len__",&wrapper_type::__len__)
        .def("__iter__",&wrapper_type::__iter__,__group_childs_docstr)
        .def("increment",&wrapper_type::increment)
        .def("remove",&wrapper_type::remove)
#if PY_MAJOR_VERSION >= 3
        .def("__next__",&wrapper_type::next)
#else
        .def("next",&wrapper_type::next)
#endif
        .add_property("filename",&wrapper_type::filename,nxgroup_filename_doc.c_str())
        .add_property("name",&wrapper_type::name,nxgroup_name_doc.c_str())
        .add_property("parent",&wrapper_type::parent,nxgroup_parent_doc.c_str())
        .add_property("size",&wrapper_type::__len__,nxgroup_size_doc.c_str())
        .def_readonly("attributes",&wrapper_type::attributes,nxgroup_attributes_doc.c_str())
        .add_property("recursive",&wrapper_type::recursive,nxgroup_recursive_doc.c_str())
        .add_property("path",&wrapper_type::path,nxgroup_path_doc.c_str())
        ;
#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif
}

