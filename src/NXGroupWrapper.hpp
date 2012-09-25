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
 * Definition of the NXGroupWrapper template.
 *
 * Created on: Feb 17, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __NXGROUPWRAPPER_HPP__
#define __NXGROUPWRAPPER_HPP__

#include <pni/nx/NXObjectType.hpp>
#include <pni/utils/service.hpp>

#include "NXWrapperHelpers.hpp"
#include "NXObjectMap.hpp"
#include "NXObjectWrapper.hpp"
#include "NXFieldWrapper.hpp"
#include "FieldCreator.hpp"
#include "ChildIterator.hpp"
#include "AttributeIterator.hpp"

/*! 
\ingroup wrappers  
\brief class tempalte for NXGroup wrapper

Class template to create wrappers for NXGroup types.
*/
template<typename GType> class NXGroupWrapper:public NXObjectWrapper<GType>
{
    public:
        typedef NXFieldWrapper<typename NXObjectMap<GType>::FieldType> field_type;
        //================constructors and destructor==========================
        //! default constructor
        NXGroupWrapper():NXObjectWrapper<GType>(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXGroupWrapper(const NXGroupWrapper<GType> &o):
            NXObjectWrapper<GType>(o)
        { }

        //----------------------------------------------------------------------
        //! move constructor
        NXGroupWrapper(NXGroupWrapper<GType> &&o):
            NXObjectWrapper<GType>(std::move(o))
        { }

        //----------------------------------------------------------------------
        //! conversion copy constructor
        explicit NXGroupWrapper(const GType &g):NXObjectWrapper<GType>(g){}

        //----------------------------------------------------------------------
        //! conversion move constructor
        explicit NXGroupWrapper(GType &&g):NXObjectWrapper<GType>(std::move(g)){}

        //----------------------------------------------------------------------
        //! destructor
        virtual ~NXGroupWrapper()
        { }

        //====================assignment operators==============================
        //! conversion copy assignment from wrapped object
        NXGroupWrapper<GType> &operator=(const GType &g)
        {
            NXObjectWrapper<GType>::operator=(g);
            return *this;
        }

        //-----------------------------------------------------------------------
        //! conversion move assignment from wrapped object
        NXGroupWrapper<GType> &operator=(GType &&g)
        {
            NXObjectWrapper<GType>::operator=(std::move(g));
            return *this;
        }

        //-----------------------------------------------------------------------
        //copy assignment 
        NXGroupWrapper<GType> &operator=(const NXGroupWrapper<GType> &o)
        {
            if(this != &o) NXObjectWrapper<GType>::operator=(o);
            return *this;
        }

        //-----------------------------------------------------------------------
        //move assignment
        NXGroupWrapper<GType> &operator=(NXGroupWrapper<GType> &&o)
        {
            if(this != &o) NXObjectWrapper<GType>::operator=(std::move(o));
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
        NXGroupWrapper<typename NXObjectMap<GType>::GroupType >
            create_group(const String &n,const String &nxclass=String()) const
        {
            typedef typename NXObjectMap<GType>::GroupType GroupType;
            NXGroupWrapper<GroupType> g(this->_object.create_group(n,nxclass));
            return g;
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
        \param name name of the new field
        \param type_code numpy type code 
        \param shape sequence object with the shape of the field
        \param chunk sequence object for the chunk shape
        \param filter a filter object for data compression
        */
        field_type create_field_nofilter(const String &name,const String
                &type_code,const object &shape=list(),const object
                &chunk=list(),const object &filter=object()) const
        {
            FieldCreator<field_type> creator(name,
                                             List2Container<shape_t>(list(shape)),
                                             List2Container<shape_t>(list(chunk)),
                                             filter);
            return creator.create(this->_object,type_code);
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
        object open_by_name(const String &n) const
        {
            typedef typename NXObjectMap<GType>::ObjectType ObjectType;
            typedef typename NXObjectMap<GType>::GroupType GroupType;
            typedef typename NXObjectMap<GType>::FieldType FieldType;
            typedef NXFieldWrapper<FieldType> field_wrapper;
            typedef NXGroupWrapper<GroupType> group_wrapper;
           
            //open the NXObject 
            ObjectType nxobject = this->_object.open(n);

            object o;

            //we use here copy construction thus we do not have to care
            //of the original nxobject goes out of scope and gets destroyed.
            if(nxobject.object_type() == pni::nx::NXObjectType::NXFIELD)
                o = object(new field_wrapper(FieldType(nxobject)));

            if(nxobject.object_type() == pni::nx::NXObjectType::NXGROUP)
                o = object(new group_wrapper(GroupType(nxobject)));

            nxobject.close();
            //this here is to avoid compiler warnings
            return o;

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
            typedef typename NXObjectMap<GType>::ObjectType ObjectType;
            ObjectType nxobject = this->_object.open(i);

            return open_by_name(nxobject.path());
        }

        //--------------------------------------------------------------------------
        /*! \brief check for objects existance
        
        Returns true if the object defined by path 'n'. 
        \return true if object exists, false otherwise
        */
        bool exists(const String &n) const
        {
            return this->_object.exists(n);
        }

        //--------------------------------------------------------------------------
        /*! \brief number of child objects

        Return the number of child objects linked below this group.
        \return number of child objects
        */
        size_t nchilds() const
        {
            return this->_object.nchilds();
        }


        //---------------------------------------------------------------------
        /*! \brief create links

        Exposes only one of the three link creation methods from the original
        NXGroup object.
        */
        void link(const String &p,const String &n) const
        {
            this->_object.link(p,n);
        }

        //----------------------------------------------------------------------
        /*! \brief get child iterator

        Returns an iterator over all child objects linked below this group.
        \return instance of ChildIterator
        */
        ChildIterator<NXGroupWrapper<GType>,object> get_child_iterator() const
        {
            return ChildIterator<NXGroupWrapper<GType>,object>(*this);
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
template<typename GTYPE> void wrap_nxgroup(const String &class_name)
{
    typedef NXGroupWrapper<GTYPE> group_wrapper;
    typedef class_<group_wrapper,bases<NXObjectWrapper<GTYPE> > > group_class;

    
    group_class(class_name.c_str())
        .def(init<>())
        .def("open",&group_wrapper::open_by_name,__group_open_docstr)
        .def("__getitem__",&group_wrapper::open)
        .def("__getitem__",&group_wrapper::open_by_name)
        .def("create_group",&group_wrapper::create_group,
                ("n",arg("nxclass")=String()),__group_create_group_docstr)
        .def("create_field",&group_wrapper::create_field_nofilter,
                ("name","type_code",arg("shape")=list(),arg("chunk")=list(),
                 arg("filter")=object()),__group_create_field_docstr)
        .def("exists",&group_wrapper::exists,__group_exists_docstr)
        .def("link",&group_wrapper::link,__group_link_docstr)
        .add_property("nchilds",&group_wrapper::nchilds,__group_nchilds_docstr)   
        .add_property("childs",&group_wrapper::get_child_iterator,__group_childs_docstr)
        ;
}


#endif
