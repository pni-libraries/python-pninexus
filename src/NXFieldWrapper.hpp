#ifndef __NXFIELDWRAPPER_HPP__
#define __NXFIELDWRAPPER_HPP__

#include "NXObjectWrapper.hpp"
#include "NXWrapperHelpers.hpp"

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

#define WRITE_SCALAR_FIELD(typeid,type)\
        if(this->_object.type_id() == typeid){\
            type value = extract<type>(o);\
            this->_object.write(value);\
        }
        //---------------------------------------------------------------------
        //! writing data to the field
        void write(const object &o) const
        {
            if(this->_object.shape().size() == 1){
                //scalar field - here we can use any scalar type to write data
                WRITE_SCALAR_FIELD(TypeID::UINT8,UInt8);
                WRITE_SCALAR_FIELD(TypeID::INT8,Int8);
                WRITE_SCALAR_FIELD(TypeID::UINT16,UInt16);
                WRITE_SCALAR_FIELD(TypeID::INT16,Int16);
                WRITE_SCALAR_FIELD(TypeID::UINT32,UInt32);
                WRITE_SCALAR_FIELD(TypeID::INT32,Int32);
                WRITE_SCALAR_FIELD(TypeID::UINT64,UInt64);
                WRITE_SCALAR_FIELD(TypeID::INT64,Int64);

                WRITE_SCALAR_FIELD(TypeID::FLOAT32,Float32);
                WRITE_SCALAR_FIELD(TypeID::FLOAT64,Float64);
                WRITE_SCALAR_FIELD(TypeID::FLOAT128,Float128);
                WRITE_SCALAR_FIELD(TypeID::COMPLEX32,Complex32);
                WRITE_SCALAR_FIELD(TypeID::COMPLEX64,Complex64);
                WRITE_SCALAR_FIELD(TypeID::COMPLEX128,Complex128);
            }else{
                //multidimensional field - the input must be a numpy array
                //check if the passed object is a numpy array
                if(!PyArray_CheckExact(o.ptr())){
                    std::cerr<<"Object is not a numpy array!"<<std::endl;
                    //need to raise an exception here
                    return;
                }

                switch(PyArray_TYPE(o.ptr())){
                    case NPY_UBYTE:
                        this->_object.write(Numpy2RefArray<UInt8>(o));
                        break;
                    case NPY_BYTE:
                        this->_object.write(Numpy2RefArray<Int8>(o));
                        break;
                    case NPY_USHORT:
                        this->_object.write(Numpy2RefArray<UInt16>(o));
                        break;
                    case NPY_SHORT:
                        this->_object.write(Numpy2RefArray<Int16>(o));
                        break;
                    case NPY_UINT:
                        this->_object.write(Numpy2RefArray<UInt32>(o));
                        break;
                    case NPY_INT:
                        this->_object.write(Numpy2RefArray<Int32>(o));
                        break;
                    case NPY_ULONG:
                        this->_object.write(Numpy2RefArray<UInt64>(o));
                        break;
                    case NPY_LONG:
                        this->_object.write(Numpy2RefArray<Int64>(o));
                        break;
                    case NPY_FLOAT:
                        this->_object.write(Numpy2RefArray<Float32>(o));
                        break;
                    case NPY_DOUBLE:
                        this->_object.write(Numpy2RefArray<Float64>(o));
                        break;
                    case NPY_LONGDOUBLE:
                        this->_object.write(Numpy2RefArray<Float128>(o));
                        break;
                    case NPY_CFLOAT:
                        this->_object.write(Numpy2RefArray<Complex32>(o));
                        break;
                    case NPY_CDOUBLE:
                        this->_object.write(Numpy2RefArray<Complex64>(o));
                        break;
                    case NPY_CLONGDOUBLE:
                        this->_object.write(Numpy2RefArray<Complex128>(o));
                        break;
                    default:
                        std::cerr<<"Array is of unkown type!"<<std::endl;

                };

            }

        }

#define READ_SCALAR_FIELD(typeid,type)\
        if(this->_object.type_id() == typeid){\
            type value;\
            this->_object.read(value);\
            object o(value);\
            return o;\
        }

#define READ_ARRAY_FIELD(typeid,type)\
        if(this->_object.type_id() == typeid){\
                PyObject *ptr =\
                    CreateNumpyArray<type>(this->_object.shape());\
                handle<> h(ptr);\
                object o(h);\
                Array<type,RefBuffer> rarray = Numpy2RefArray<type>(o);\
                this->_object.read(rarray);\
                return o;\
        }
        //---------------------------------------------------------------------
        //! reading data from the field
        object read() const
        {
            if(this->_object.shape().size() == 1){
                //the field contains only a single value - can return a
                //primitive python object
                
                READ_SCALAR_FIELD(TypeID::UINT8,UInt8);
                READ_SCALAR_FIELD(TypeID::INT8,Int8);
                READ_SCALAR_FIELD(TypeID::UINT16,UInt16);
                READ_SCALAR_FIELD(TypeID::INT16,Int16);
                READ_SCALAR_FIELD(TypeID::UINT32,UInt32);
                READ_SCALAR_FIELD(TypeID::INT32,Int32);
                READ_SCALAR_FIELD(TypeID::UINT64,UInt64);
                READ_SCALAR_FIELD(TypeID::INT64,Int64);

                READ_SCALAR_FIELD(TypeID::FLOAT32,Float32);
                READ_SCALAR_FIELD(TypeID::FLOAT64,Float64);
                READ_SCALAR_FIELD(TypeID::FLOAT128,Float128);
                READ_SCALAR_FIELD(TypeID::COMPLEX32,Complex32);
                READ_SCALAR_FIELD(TypeID::COMPLEX64,Complex64);
                READ_SCALAR_FIELD(TypeID::COMPLEX128,Complex128);
                
                READ_SCALAR_FIELD(TypeID::STRING,String);

            }else{
                //the field contains multidimensional data  - we return a numpy
                //array

                READ_ARRAY_FIELD(TypeID::UINT8,UInt8);
                READ_ARRAY_FIELD(TypeID::INT8,Int8);
                READ_ARRAY_FIELD(TypeID::UINT16,UInt16);
                READ_ARRAY_FIELD(TypeID::INT16,Int16);
                READ_ARRAY_FIELD(TypeID::UINT32,UInt32);
                READ_ARRAY_FIELD(TypeID::INT32,Int32);
                READ_ARRAY_FIELD(TypeID::UINT64,UInt64);
                READ_ARRAY_FIELD(TypeID::INT64,Int64);

                READ_ARRAY_FIELD(TypeID::FLOAT32,Float32);
                READ_ARRAY_FIELD(TypeID::FLOAT64,Float64);
                READ_ARRAY_FIELD(TypeID::FLOAT128,Float128);
                READ_ARRAY_FIELD(TypeID::COMPLEX32,Complex32);
                READ_ARRAY_FIELD(TypeID::COMPLEX64,Complex64);
                READ_ARRAY_FIELD(TypeID::COMPLEX128,Complex128);
            }

            //we should raise an exception here

            //this is only to avoid compiler warnings
            return object();
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
        ;
}

#endif
