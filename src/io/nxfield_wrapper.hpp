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
// Created on: March 8, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <pni/core/types.hpp>
#include <pni/io/exceptions.hpp>
#include <pni/io/nx/nxobject_traits.hpp>
#include <pni/io/nx/algorithms/get_path.hpp>

#include <core/utils.hpp>
#include <core/numpy_utils.hpp>

#include "nxattribute_manager_wrapper.hpp"
#include "utils.hpp"
#include "io_operations.hpp"

template<typename GTYPE> class nxgroup_wrapper;

//!
//! \ingroup wrappers
//! \brief NXField wrapper template
//! 
//! Template to produce wrappers for NXField types.
//!
template<typename FIELDT> class nxfield_wrapper
{
    public:
        typedef FIELDT field_type;
        
        static const pni::io::nx::nximp_code imp_id = 
                     pni::io::nx::nximp_code_map<field_type>::icode;
        typedef nxfield_wrapper<field_type> wrapper_type;
        typedef typename pni::io::nx::nxobject_trait<imp_id>::group_type group_type;
        typedef nxgroup_wrapper<group_type> group_wrapper_type;

        typedef decltype(field_type::attributes) attribute_manager_type;
        typedef nxattribute_manager_wrapper<attribute_manager_type>
            attribute_manager_wrapper;
    private:
        field_type _field;
    public:
        //=============constrcutors and destructor=============================
        //! default constructor
        nxfield_wrapper():
            _field(),
            attributes(_field.attributes)
        {}

        //---------------------------------------------------------------------
        //! copy constructor
        nxfield_wrapper(const wrapper_type &f):
            _field(f._field),
            attributes(_field.attributes)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        nxfield_wrapper(wrapper_type &&f):
            _field(std::move(f._field)),
            attributes(_field.attributes)
        {}

        //--------------------------------------------------------------------
        //! copy constructor from wrapped type
        explicit nxfield_wrapper(const field_type &o):
            _field(o),
            attributes(_field.attributes)
        {}

        //!-------------------------------------------------------------------
        //! move constructor from wrapped type
        explicit nxfield_wrapper(field_type &&o):
            _field(std::move(o)),
            attributes(_field.attributes)
        {}

        operator field_type() const
        {
            return _field;
        }

        //---------------------------------------------------------------------
        attribute_manager_wrapper attributes;

        //=================wrap some conviencen methods========================
        //!
        //! \brief get type code 
        //!   
        //! Returns the type-code of the data stored in the field as numpy 
        //! type string. The type-code will be exposed as a read-only 
        //! property.
        //!
        //! \return numpy type string
        //!
        pni::core::string type_id() const 
        { 
            return numpy::type_str(_field.type_id()); 
        }

        //---------------------------------------------------------------------
        //!
        //! \brief get field shape
        //!
        //! Return the shape of the field as tuple. The length of the tuple 
        //! is the rank of the field and its elements define the number of 
        //! elements along each dimension. The shape will be exposed as a 
        //! read-only property.
        //!
        //! \return shape as tuple
        //!
        boost::python::tuple shape() const
        {
            auto shape = _field.template shape<pni::core::shape_t>();
            return boost::python::tuple(Container2List(shape));
        }

        //---------------------------------------------------------------------
        //!
        //! \brief writing data to the field
        //!
        //! Write data to the field. The data is passed to the wrapper as a 
        //! Python object. The method tries to figure out what kind of object 
        //! it is and if it can be written to the field. 
        //!
        //! \param o object from which to write data
        //!
        void write(const boost::python::object &o) const
        {

            if(numpy::is_array(o))
                write_data(_field,o);
            else
                write_data(_field,numpy::to_numpy_array(o));
        }

        //---------------------------------------------------------------------
        //!
        //! \brief reading data from the field
        //! 
        //! Reading data from a field. The method returns a Python object 
        //! and tries to figure out by itself which kind of object and 
        //! datatype to use.  If this fails an exception will be thrown.
        //!
        //! \throws type_error if return type determination fails
        //! \return Python object with the read data
        //!
        boost::python::object read() const
        {
            using namespace boost::python;

            object np_array = read_data(_field);
    
            if(numpy::get_size(np_array)==1) np_array = get_first_element(np_array);

            return np_array;
            
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
            using namespace boost::python; 

            typedef std::vector<pni::core::slice> selection_type;

            boost::python::tuple sel;
            if(PyTuple_Check(t.ptr()))
                sel = tuple(t);
            else
                sel = make_tuple<object>(t);

            //first we need to create a selection
            selection_type selection = create_selection(sel,_field);

            object np_array = read_data(_field(selection));

            if(numpy::get_size(np_array)==1) 
                np_array = get_first_element(np_array);

            return np_array;
        }

        //---------------------------------------------------------------------
        //!
        //! \brief write data according to a selection
        //!
        //! Write data from a selection defined by tuple t. 
        //!
        //! \param t tuple with selection information
        //! \param o object with data to write.
        //!
        void __setitem__(const boost::python::object &t, 
                         const boost::python::object &o)
        {
            using namespace boost::python;

            typedef std::vector<pni::core::slice> selection_type;
            tuple sel;

            if(PyTuple_Check(t.ptr()))
                sel = tuple(t);
            else
                sel = make_tuple<object>(t);

            selection_type selection = create_selection(sel,_field);

            if(numpy::is_array(o))
                write_data(_field(selection),o);
            else
                write_data(_field(selection),numpy::to_numpy_array(o));

        }

        //----------------------------------------------------------------------
        //!
        //! \brief grow field object
        //!
        //! Grow a field object along dimension d by s elements.
        //! \param d dimension index
        //! \param s extend by which to grow the field
        //!
        void grow(size_t d=0,size_t s=1) { _field.grow(d,s); }

        //-----------------------------------------------------------------------
        //!
        //! \brief get size
        //!
        //! Return the total number of elements of the field.
        //!
        //! \return total number of elements.
        //!
        size_t size() const { return _field.size(); }
        
        //--------------------------------------------------------------------
        bool is_valid() const { return _field.is_valid(); } 

        //--------------------------------------------------------------------
        void close() { _field.close(); }

        //--------------------------------------------------------------------
        pni::core::string filename() const { return _field.filename(); }

        //--------------------------------------------------------------------
        pni::core::string name() const { return _field.name(); }

        //--------------------------------------------------------------------
        group_wrapper_type parent() const
        {
            return group_wrapper_type(group_type(_field.parent()));
        }

        //--------------------------------------------------------------------
        size_t __len__() const { return _field.size(); }
       
        //--------------------------------------------------------------------
        pni::core::string path() const 
        { 
            return pni::io::nx::get_path(_field); 
        }
        
};

//-------------------------------documentation strings-------------------------
static const char __field_dtype_docstr[] = 
"Read only property providing the datatype of the field as numpy type code";

static const char __field_shape_docstr[] = 
"Read only property providing the shape of the field as tuple";

static const char __field_grow_docstr[]=
"Grow the field \n"
"\n"
"Grow the field along dimension 'dim' by 'ext' elements.\n"
"\n"
":param long dim: dimension along which to grow\n"
":param long ext: number of elements by which to grow\n"
;

static const pni::core::string nxfield_filename_doc = 
"Read only property returning the name of the file the field belongs to\n";

static const pni::core::string nxfield_name_doc = 
"Read only property returning the name of the field\n";

static const pni::core::string nxfield_parent_doc = 
"Read only property returning the parent group of this field\n";

static const pni::core::string nxfield_size_doc = 
"Read only property returing the number of elements this field holds\n";

static const pni::core::string nxfield_attributes_doc = 
"Read only property with the attribute manager for this field\n";

static const pni::core::string nxfield_path_doc = 
"Read only property returning the NeXus path for this field\n";

static const pni::core::string nxfield_close_doc = 
"Close this field";

static const pni::core::string nxfield_is_valid_doc = 
"Read only property returning :py:const:`True` if this instance is a valid"
" NeXus object";

static const pni::core::string nxfield_read_doc = 
"Read data from field\n"
"\n"
"Read all data stored in a field and return it as a numpy array of appropriate"
"type.\n"
"\n"
":return: data stored in the field\n"
":rtype: numpy array\n"
;

static const pni::core::string nxfield_write_doc = 
"Write data to field\n"
"\n"
"Write all data stored in the *data* argument to the field. *data* must "
"be a numpy array of appropriate shape. The type of the the numpy array "
"must be convertible to the type of the field. \n"
"\n"
":param numpy.ndarray data: the data which should be written to disk\n"
":raises :py:exc:`ShapeMismatchError`: if the shape of the field does not match\n"
":raises :py:exc:`SizeMismatchError`: if the size of field and *data* do not match\n"
;


//! 
//! \ingroup wrappers
//! \brief create new NXField wrapper
//! 
//! Template function to create a wrapper for NXField type FType. 
//! \param class_name Python name of the new class
//!
template<typename FIELDT> void wrap_nxfield()
{
    using namespace boost::python;

    typedef nxfield_wrapper<FIELDT> wrapper_type; 

    class_<wrapper_type >("nxfield")
        .def(init<>())
        .add_property("dtype",&wrapper_type::type_id,__field_dtype_docstr)
        .add_property("shape",&wrapper_type::shape,__field_shape_docstr)
        .add_property("size",&wrapper_type::size,nxfield_size_doc.c_str())
        .add_property("filename",&wrapper_type::filename,nxfield_filename_doc.c_str())
        .add_property("name",&wrapper_type::name,nxfield_name_doc.c_str())
        .add_property("parent",&wrapper_type::parent,nxfield_parent_doc.c_str())
        .add_property("is_valid",&wrapper_type::is_valid,nxfield_is_valid_doc.c_str())
        .add_property("path",&wrapper_type::path,nxfield_path_doc.c_str())
        .def("write",&wrapper_type::write,nxfield_write_doc.c_str())
        .def("read",&wrapper_type::read,nxfield_read_doc.c_str())
        .def("__getitem__",&wrapper_type::__getitem__)
        .def("__setitem__",&wrapper_type::__setitem__)
        .def("grow",&wrapper_type::grow,(arg("dim")=0,arg("ext")=1),__field_grow_docstr)
        .def("close",&wrapper_type::close,nxfield_close_doc.c_str())
        .def_readonly("attributes",&wrapper_type::attributes,nxfield_attributes_doc.c_str())
        ;
}

