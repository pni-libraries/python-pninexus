#ifndef __NXFIELDWRAPPER_HPP__
#define __NXFIELDWRAPPER_HPP__

#include <boost/python/slice.hpp>
#include "NXObjectWrapper.hpp"
#include "NXWrapperHelpers.hpp"
#include "NXIOOperations.hpp"

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
        NXFieldWrapper<FieldT> &operator=(NXFieldWrapper<FieldT> &&o)
        {
            if(this != &o) NXObjectWrapper<FieldT>::operator=(std::move(o));
            return *this;
        }

        //=================wrap some conviencen methods========================
        //! get the type string of the field
        String type_id() const
        {
            return typeid2str(this->_object.type_id());
        }

        //---------------------------------------------------------------------
        //! get the shape as list object
        object shape() const
        {
            return Shape2List(this->_object.shape());
        }

        //---------------------------------------------------------------------
        //! writing data to the field
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
        //! reading data from the field
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

            //we should raise an exception here

            //this is only to avoid compiler warnings
            return object();
        }
       
        //---------------------------------------------------------------------
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
        object __getitem__object(const object &o)
        {
            //need to check here if o is already a tuple 
            if(PyTuple_Check(o.ptr()))
                return __getitem__tuple(tuple(o));
            else
                return __getitem__tuple(make_tuple<object>(o));
        }

        //---------------------------------------------------------------------
        object __getitem__index(size_t i){
            return __getitem__tuple(make_tuple<size_t>(i));
        }

        //---------------------------------------------------------------------
        object __getitem__slice(const slice &o){
            return __getitem__tuple(make_tuple<slice>(o));
        }

        //---------------------------------------------------------------------
        void __setitem__object(const object &o,const object &d)
        {
            //need to check here if o is already a tuple 
            if(PyTuple_Check(o.ptr()))
                __setitem__tuple(tuple(o),d);
            else
                __setitem__tuple(make_tuple<object>(o),d);
        }
        //---------------------------------------------------------------------
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

        //---------------------------------------------------------------------
        void __setitem__index(size_t i,const object &o){
            __setitem__tuple(make_tuple<size_t>(i),o);
        }

        //---------------------------------------------------------------------
        void __setitem__slice(const slice &s,const object &o){
             __setitem__tuple(make_tuple<slice>(s),o);
        }
        
        //--------------------------------------------------------------------------
        void grow_default()
        {
            this->_object.grow(0,1);
        }

        void grow_dim(size_t d)
        {
            this->_object.grow(d,1);
        }

        void grow(size_t d,size_t s)
        {
            this->_object.grow(d,s);
        }
        
};


template<typename FType> void wrap_nxfield(const String &class_name)
{

    class_<NXFieldWrapper<FType>,bases<NXObjectWrapper<FType> > >(class_name.c_str())
        .def(init<>())
        .add_property("type_id",&NXFieldWrapper<FType>::type_id)
        .add_property("shape",&NXFieldWrapper<FType>::shape)
        .def("write",&NXFieldWrapper<FType>::write)
        .def("read",&NXFieldWrapper<FType>::read)
        .def("__getitem__",&NXFieldWrapper<FType>::__getitem__index)
        .def("__getitem__",&NXFieldWrapper<FType>::__getitem__slice)
        .def("__getitem__",&NXFieldWrapper<FType>::__getitem__tuple)
        .def("__getitem__",&NXFieldWrapper<FType>::__getitem__object)
        .def("__setitem__",&NXFieldWrapper<FType>::__setitem__index)
        .def("__setitem__",&NXFieldWrapper<FType>::__setitem__slice)
        .def("__setitem__",&NXFieldWrapper<FType>::__setitem__tuple)
        .def("__setitem__",&NXFieldWrapper<FType>::__setitem__object)
        .def("grow",&NXFieldWrapper<FType>::grow)
        .def("grow",&NXFieldWrapper<FType>::grow_default)
        .def("grow",&NXFieldWrapper<FType>::grow_dim)
        ;
}

#endif
