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
 * Definition of the wrapper template for NXField classes.
 *
 * Created on: March 8, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */
#ifndef __NXFIELDWRAPPER_HPP__
#define __NXFIELDWRAPPER_HPP__

#include <boost/python/slice.hpp>
#include "NXObjectWrapper.hpp"
#include "NXWrapperHelpers.hpp"
#include "NXIOOperations.hpp"

/*!
\ingroup wrappers
\brief NXField wrapper template

Template to produce wrappers for NXField types.
*/
template<typename FieldT> class NXFieldWrapper:
    public NXObjectWrapper<FieldT>
{
    public:
        //=============constrcutors and destructor=============================
        //! default constructor
        NXFieldWrapper():NXObjectWrapper<FieldT>(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXFieldWrapper(const NXFieldWrapper<FieldT> &f):
            NXObjectWrapper<FieldT>(f)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        NXFieldWrapper(NXFieldWrapper<FieldT> &&f):
            NXObjectWrapper<FieldT>(std::move(f))
        {}

        //--------------------------------------------------------------------
        //! copy constructor from wrapped type
        explicit NXFieldWrapper(const FieldT &o):NXObjectWrapper<FieldT>(o)
        {}

        //!-------------------------------------------------------------------
        //! move constructor from wrapped type
        explicit NXFieldWrapper(FieldT &&o):
            NXObjectWrapper<FieldT>(std::move(o))
        {}

        //---------------------------------------------------------------------
        //! destructor
        ~NXFieldWrapper(){}
            

        //=========================assignment operators========================
        //! copy assignment from wrapped type
        NXFieldWrapper<FieldT> &operator=(const FieldT &f)
        {
            NXObjectWrapper<FieldT>::operator=(f);
            return *this;
        }

        //---------------------------------------------------------------------
        //! move assignment from wrapped type
        NXFieldWrapper<FieldT> &operator=(FieldT &&f)
        {
            NXObjectWrapper<FieldT>::operator=(std::move(f));
            return *this;
        }

        //---------------------------------------------------------------------
        //! copy assignment operator
        NXFieldWrapper<FieldT> &operator=(const NXFieldWrapper<FieldT> &o)
        {
            if(this != &o) NXObjectWrapper<FieldT>::operator=(o);
            return *this;
        }

        //---------------------------------------------------------------------
        //! move assignment
        NXFieldWrapper<FieldT> &operator=(NXFieldWrapper<FieldT> &&o)
        {
            if(this != &o) NXObjectWrapper<FieldT>::operator=(std::move(o));
            return *this;
        }

        //=================wrap some conviencen methods========================
        /*! \brief get type code 
          
        Returns the type-code of the data stored in the field as numpy type
        string. The type-code will be exposed as a read-only property.
        \return numpy type string
        */
        String type_id() const
        {
            return typeid2str(this->_object.type_id());
        }

        //---------------------------------------------------------------------
        /*! \brief get field shape

        Return the shape of the field as tuple. The length of the tuple is the 
        rank of the field and its elements define the number of elements along 
        each dimension. The shape will be exposed as a read-only property.
        \return shape as tuple
        */
        tuple shape() const
        {
            return tuple(Shape2List(this->_object.shape()));
        }

        //---------------------------------------------------------------------
        /*! \brief writing data to the field

        Write data to the field. The data is passed to the wrapper as a Python
        object. The method tries to figure out what kind of object it is 
        and if it can be written to the field. 
        \param o object from which to write data
        */
        void write(const object &o) const
        {
            if(this->_object.shape().size() == 1){
                //scalar field - here we can use any scalar type to write data
                io_write<ScalarWriter>(this->_object,o);
            }else{
                //multidimensional field - the input must be a numpy array
                //check if the passed object is a numpy array

                ArrayWriter::write(this->_object,o);

            }
        }

        //---------------------------------------------------------------------
        /*! \brief reading data from the field

        Reading data from a field. The method returns a Python object and tries
        to figure out by itself which kind of object and datatype to use. 
        If this fails an exception will be thrown.
        \throws TypeError if return type determination fails
        \return Python object with the read data
        */
        object read() const
        {
            if(this->_object.shape().size() == 1){
                //the field contains only a single value - can return a
                //primitive python object
                return io_read<ScalarReader>(this->_object);                

            }else{
                //the field contains multidimensional data  - we return a numpy
                //array
                return io_read<ArrayReader>(this->_object);
            }

            TypeError error;
            error.issuer("template<typename FielT> object NXFieldWrapper"
                    "<FieldT>::read() const");
            error.description("Cannot determine return type!");
            throw(error);


            //this is only to avoid compiler warnings
            return object();
        }
       
        //---------------------------------------------------------------------
        /*! \brief the core __getitem__ implementation

        The most fundamental implementation of the __getitem__ method. 
        The tuple passed to the method can contain indices, slices, and a single
        ellipsis. This method is doing the real work - all other __getitem__ 
        implementations simply call this method.
        \param t tuple with 
        */
        object __getitem__tuple(const tuple &t){

            //first we need to create a selection
            NXSelection selection = create_selection(t,this->_object);

            //once the selection is build we can start to create the 
            //return value
            if(selection.size()==1){
                //in this case we return a primitive python value
                return io_read<ScalarReader>(selection);
            }else{
                //a numpy array will be returned
                return io_read<ArrayReader>(selection);
            }


            return object();

        }
        //---------------------------------------------------------------------
        /*! \brief __getitem__ entry method

        This method is called when a user invokes the __getitem__ method on a
        NXField object in python. The method converts the object that is passed
        as input argument to a tuple if necessary and than passes this to the 
        __getitem__tuple method. 
        \param o Python object describing the selection
        \return Python object with the data from the selection
        */
        object __getitem__(const object &o)
        {
            //need to check here if o is already a tuple 
            if(PyTuple_Check(o.ptr()))
                return __getitem__tuple(tuple(o));
            else
                return __getitem__tuple(make_tuple<object>(o));
        }

        //---------------------------------------------------------------------
        /*! \brief __setitem__ implementation

        As for __getitem__ this method is called if a user invokes the
        __setitem__ method on an NXField object in Python. The method converts
        the object passed as input argument to a tuple if necessary and then
        moves on to __setitem__tuple.
        \param o selection object
        \param d Python object holding the data
        */
        void __setitem__(const object &o,const object &d)
        {
            //need to check here if o is already a tuple 
            if(PyTuple_Check(o.ptr()))
                __setitem__tuple(tuple(o),d);
            else
                __setitem__tuple(make_tuple<object>(o),d);
        }
        //---------------------------------------------------------------------
        /*! \brief write data according to a selection

        Write data from a selection defined by tuple t. 
        \param t tuple with selection information
        \param o object with data to write.
        */
        void __setitem__tuple(const tuple &t,const object &o){
            
            NXSelection selection = create_selection(t,this->_object);

            if(selection.shape().size() == 1){
                //in this case we can write only a single scalar value. Thus the
                //object passed must be a simple scalar value
                io_write<ScalarWriter>(selection,o);
            }else{
                //here we have two possibl: 
                //1.) object referes to a scalar => all positions marked by the 
                //    selection will be set to this scalar value
                //2.) object referes to an array => if the selection shape and
                //    the array shape match (sizes match) we can write array
                //    data.

                //let us assume here that we only do broadcast
                if(!PyArray_CheckExact(o.ptr())){
                    ArrayBroadcastWriter::write(selection,o);
                }else{
                    ArrayWriter::write(selection,o);
                }
            }
        }

        //--------------------------------------------------------------------------
        /*! \brief grow field object

        Grow a field object along dimension d by s elements.
        \param d dimension index
        \param s extend by which to grow the field
        */
        void grow(size_t d=0,size_t s=1)
        {
            this->_object.grow(d,s);
        }
        
};

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

/*! 
\ingroup wrappers
\brief create new NXField wrapper

Template function to create a wrapper for NXField type FType. 
\param class_name Python name of the new class
*/
template<typename FType> void wrap_nxfield(const String &class_name)
{

    class_<NXFieldWrapper<FType>,bases<NXObjectWrapper<FType> > >(class_name.c_str())
        .def(init<>())
        .add_property("dtype",&NXFieldWrapper<FType>::type_id,__field_dtype_docstr)
        .add_property("shape",&NXFieldWrapper<FType>::shape,__field_shape_docstr)
        .def("write",&NXFieldWrapper<FType>::write)
        .def("read",&NXFieldWrapper<FType>::read)
        .def("__getitem__",&NXFieldWrapper<FType>::__getitem__)
        .def("__setitem__",&NXFieldWrapper<FType>::__setitem__)
        .def("grow",&NXFieldWrapper<FType>::grow,(arg("dim")=0,arg("ext")=1),__field_grow_docstr)
        ;
}

#endif
