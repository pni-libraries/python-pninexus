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

#include <boost/python/slice.hpp>
#include <pni/io/exceptions.hpp>
#include <pni/io/nx/nxobject_traits.hpp>
#include "nxwrapper_utils.hpp"
#include "nxio_operations.hpp"
#include "numpy_utils.hpp"

using namespace pni::io::nx;

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
        
        static const nximp_code imp_id = nximp_code_map<field_type>::icode;
        typedef nxfield_wrapper<field_type> wrapper_type;
        typedef typename nxobject_trait<imp_id>::group_type group_type;
        typedef nxgroup_wrapper<group_type> group_wrapper_type;
    private:
        field_type _field;
    public:
        //=============constrcutors and destructor=============================
        //! default constructor
        nxfield_wrapper(){}

        //---------------------------------------------------------------------
        //! copy constructor
        nxfield_wrapper(const wrapper_type &f):_field(f._field) {}

        //---------------------------------------------------------------------
        //! move constructor
        nxfield_wrapper(wrapper_type &&f):_field(std::move(f._field)) {}

        //--------------------------------------------------------------------
        //! copy constructor from wrapped type
        explicit nxfield_wrapper(const field_type &o):_field(o) {}

        //!-------------------------------------------------------------------
        //! move constructor from wrapped type
        explicit nxfield_wrapper(field_type &&o):_field(std::move(o))
        {}

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
        string type_id() const 
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
        tuple shape() const
        {
            auto shape = _field.template shape<shape_t>();
            return tuple(Container2List(shape));
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
        void write(const object &o) const
        {
            if(is_scalar(o))
            {
                //write a scalar to the field
                //scalar field - here we can use any scalar type to write data
                io_write<scalar_writer>(_field,o);
            }
            else if(numpy::is_array(o))
            {
                //write a numpy array to the field
                //multidimensional field - the input must be a numpy array
                //check if the passed object is a numpy array
                io_write<array_writer>(_field,o);
            }
            else
                throw type_error(EXCEPTION_RECORD,
                        "Argument must be either a numpy array or a "
                        "Python scalar!");
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
        object read() const
        {
            if(_field.size() == 1)
            {
                //the field contains only a single value - can return a
                //primitive python object
                return io_read<scalar_reader>(_field);                
            }
            else
            {
                //the field contains multidimensional data  - we return a numpy
                //array
                return io_read<array_reader>(_field);
            }

            //should be rather rare
            throw type_error(EXCEPTION_RECORD,"Cannot determine return type!");

            //this is only to avoid compiler warnings
            return object();
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
        object __getitem__tuple(const tuple &t)
        {
            typedef std::vector<pni::core::slice> selection_type;

            //first we need to create a selection
            selection_type selection = create_selection(t,_field);

            //once the selection is build we can start to create the 
            //return value
            if(_field(selection).size()==1)
                //in this case we return a primitive python value
                return io_read<scalar_reader>(_field(selection));
            else
                //a numpy array will be returned
                return io_read<array_reader>(_field(selection));

            //throw an exception if we cannot handle the user request
            throw pni::io::object_error(EXCEPTION_RECORD,
                                             "cannot handle user request");

            return object(); //make the compiler happy
        }
        //---------------------------------------------------------------------
        //!
        //! \brief __getitem__ entry method
        //!
        //! This method is called when a user invokes the __getitem__ method 
        //! on a NXField object in python. The method converts the object 
        //! that is passed as input argument to a tuple if necessary and 
        //! than passes this to the __getitem__tuple method. 
        //!
        //! \param o Python object describing the selection
        //! \return Python object with the data from the selection
        //!
        object __getitem__(const object &o)
        {
            //need to check here if o is already a tuple 
            if(PyTuple_Check(o.ptr()))
                return __getitem__tuple(tuple(o));
            else
                return __getitem__tuple(make_tuple<object>(o));
        }

        //---------------------------------------------------------------------
        //!
        //! \brief __setitem__ implementation
        //!
        //! As for __getitem__ this method is called if a user invokes the
        //! __setitem__ method on an NXField object in Python. The method 
        //! converts the object passed as input argument to a tuple if 
        //! necessary and then moves on to __setitem__tuple.
        //!
        //! \param o selection object
        //! \param d Python object holding the data
        //!
        void __setitem__(const object &o,const object &d)
        {
            //need to check here if o is already a tuple 
            if(PyTuple_Check(o.ptr()))
                __setitem__tuple(tuple(o),d);
            else
                __setitem__tuple(make_tuple<object>(o),d);
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
        void __setitem__tuple(const tuple &t,const object &o)
        {
            typedef std::vector<pni::core::slice> selection_type;
            selection_type selection = create_selection(t,_field);

            if(is_scalar(o))
                io_write<scalar_writer>(_field(selection),o);
            else if(numpy::is_array(o))
                io_write<array_writer>(_field(selection),o);
            else
                throw type_error(EXCEPTION_RECORD,
                        "Object must be either a numpy arry or a "
                        "python scalar!");
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
        string filename() const { return _field.filename(); }

        //--------------------------------------------------------------------
        string name() const { return _field.name(); }

        //--------------------------------------------------------------------
        group_wrapper_type parent() const
        {
            return group_wrapper_type(group_type(_field.parent()));
        }

        //--------------------------------------------------------------------
        size_t __len__() const { return _field.size(); }
        
};

//-------------------------------documentation strings-------------------------
static const char __field_dtype_docstr[] = 
"Read only property providing the datatype of the field as numpy type code";

static const char __field_shape_docstr[] = 
"Read only property providing the shape of the field as tuple";

static const char __field_grow_docstr[]=
"Grow the field along dimension 'dim' by 'ext' elements.\n\n"
"Required input arguments:\n"
"\tdim .............. dimension along which to grow\n"
"\text .............. number of elements by which to grow\n"
;

static const char __field_size_docstr[] = "total number of elements in the field\n";

//! 
//! \ingroup wrappers
//! \brief create new NXField wrapper
//! 
//! Template function to create a wrapper for NXField type FType. 
//! \param class_name Python name of the new class
//!
template<typename FIELDT> void wrap_nxfield()
{
    typedef nxfield_wrapper<FIELDT> wrapper_type; 

    class_<wrapper_type >("nxfield")
        .def(init<>())
        .add_property("dtype",&wrapper_type::type_id,__field_dtype_docstr)
        .add_property("shape",&wrapper_type::shape,__field_shape_docstr)
        .add_property("size",&wrapper_type::size,__field_size_docstr)
        .add_property("filename",&wrapper_type::filename)
        .add_property("name",&wrapper_type::name)
        .add_property("parent",&wrapper_type::parent)
        .def("write",&wrapper_type::write)
        .def("read",&wrapper_type::read)
        .def("__getitem__",&wrapper_type::__getitem__)
        .def("__setitem__",&wrapper_type::__setitem__)
        .def("grow",&wrapper_type::grow,(arg("dim")=0,arg("ext")=1),__field_grow_docstr)
        .def("close",&wrapper_type::close)
        .def("is_valid",&wrapper_type::is_valid)
        ;
}

