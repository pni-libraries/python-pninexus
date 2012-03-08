#ifndef __NXGROUPWRAPPER_HPP__
#define __NXGROUPWRAPPER_HPP__

#include <pni/nx/NXObjectType.hpp>

#include "NXWrapperHelpers.hpp"
#include "NXObjectMap.hpp"
#include "NXObjectWrapper.hpp"
#include "NXFieldWrapper.hpp"


template<typename GType> class NXGroupWrapper:public NXObjectWrapper<GType>
{
    public:
        //================constructors and destructor==========================
        //! default constructor
        NXGroupWrapper():NXObjectWrapper<GType>(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXGroupWrapper(const NXGroupWrapper<GType> &o):
            NXObjectWrapper<GType>(o){
        }

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

        //------------------------------------------------------------------------
        //! create a group
        NXGroupWrapper<typename NXObjectMap<GType>::GroupType > create_group(const String &n) const
        {
            typedef typename NXObjectMap<GType>::GroupType GroupType;
            NXGroupWrapper<GroupType> g(this->_object.create_group(n));
            return g;
        }

#define CREATE_SCALAR_FIELD(typecode,type)\
        if(type_code == typecode)\
            return field_type(this->_object.template create_field<type>(n));

        //-------------------------------------------------------------------------
        //! create a scalar field field
        NXFieldWrapper<typename NXObjectMap<GType>::FieldType>
            create_scalar_field(const String &n,const String type_code)
        {
            typedef NXFieldWrapper<typename NXObjectMap<GType>::FieldType> field_type;

            CREATE_SCALAR_FIELD("uint8",UInt8);
            CREATE_SCALAR_FIELD("int8",Int8);
            CREATE_SCALAR_FIELD("uint16",UInt16);
            CREATE_SCALAR_FIELD("int16",Int16);
            CREATE_SCALAR_FIELD("uint32",UInt32);
            CREATE_SCALAR_FIELD("int32",Int32);
            CREATE_SCALAR_FIELD("uint64",UInt64);
            CREATE_SCALAR_FIELD("int64",Int64);

            CREATE_SCALAR_FIELD("float32",Float32);
            CREATE_SCALAR_FIELD("float64",Float64);
            CREATE_SCALAR_FIELD("float128",Float128);
            
            CREATE_SCALAR_FIELD("complex64",Complex32);
            CREATE_SCALAR_FIELD("complex128",Complex64);
            CREATE_SCALAR_FIELD("complex256",Complex128);

            CREATE_SCALAR_FIELD("string",String);

            //should raise an exception here

            //this here is only to avoid compiler warnings
            return field_type();
        }

#define CREATE_ARRAY_FIELD(typecode,type)\
        if(type_code == typecode)\
            return field_type(this->_object.template create_field<type>(n,s));
        //-------------------------------------------------------------------------
        //! create a multidimensional field
        NXFieldWrapper<typename NXObjectMap<GType>::FieldType>
            create_array_field(const String &n,const String &type_code,const
                    list &shape)
        {
            typedef NXFieldWrapper<typename NXObjectMap<GType>::FieldType> field_type;
            //create the shape of the new field
            Shape s = List2Shape(shape);

            CREATE_ARRAY_FIELD("uint8",UInt8);
            CREATE_ARRAY_FIELD("int8",Int8);
            CREATE_ARRAY_FIELD("uint16",UInt16);
            CREATE_ARRAY_FIELD("int16",Int16);
            CREATE_ARRAY_FIELD("uint32",UInt32);
            CREATE_ARRAY_FIELD("int32",Int32);
            CREATE_ARRAY_FIELD("uint64",UInt64);
            CREATE_ARRAY_FIELD("int64",Int64);

            CREATE_ARRAY_FIELD("float32",Float32);
            CREATE_ARRAY_FIELD("float64",Float64);
            CREATE_ARRAY_FIELD("float128",Float128);
            
            CREATE_ARRAY_FIELD("complex64",Complex32);
            CREATE_ARRAY_FIELD("complex128",Complex64);
            CREATE_ARRAY_FIELD("complex256",Complex128);

            //should raise an exception here

            //this here is only to avoid compiler warning
            return field_type();
        }

#define CREATE_ARRAY_FIELD_CHUNKED(typecode,type)\
        if(type_code == typecode)\
            return field_type(this->_object.template create_field<type>(n,s,cs));

        //-------------------------------------------------------------------------
        //! create a multidimensional field with chunks
        NXFieldWrapper<typename NXObjectMap<GType>::FieldType>
            create_array_field_chunked(const String &n,const String &type_code,
                    const list &shape,const list &chunk)
        {
            typedef NXFieldWrapper<typename NXObjectMap<GType>::FieldType> field_type;
            //create the shape of the new array
            Shape s = List2Shape(shape);
            //create the chunk shape
            Shape cs = List2Shape(chunk);

            CREATE_ARRAY_FIELD_CHUNKED("uint8",UInt8);
            CREATE_ARRAY_FIELD_CHUNKED("int8",Int8);
            CREATE_ARRAY_FIELD_CHUNKED("uint16",UInt16);
            CREATE_ARRAY_FIELD_CHUNKED("int16",Int16);
            CREATE_ARRAY_FIELD_CHUNKED("uint32",UInt32);
            CREATE_ARRAY_FIELD_CHUNKED("int32",Int32);
            CREATE_ARRAY_FIELD_CHUNKED("uint64",UInt64);
            CREATE_ARRAY_FIELD_CHUNKED("int64",Int64);

            CREATE_ARRAY_FIELD_CHUNKED("float32",Float32);
            CREATE_ARRAY_FIELD_CHUNKED("float64",Float64);
            CREATE_ARRAY_FIELD_CHUNKED("float128",Float128);
            
            CREATE_ARRAY_FIELD_CHUNKED("complex64",Complex32);
            CREATE_ARRAY_FIELD_CHUNKED("complex128",Complex64);
            CREATE_ARRAY_FIELD_CHUNKED("complex256",Complex128);

            //should raise an exception here

            //this here is to avoid compiler warnings
            return field_type();
        }

        //-------------------------------------------------------------------------
        /*! \brief open an object

        This method opens an object and tries to figure out by itself what kind
        of object is has to deal with. Consequently it returns already a Python
        obeject of the appropriate type. 
        \param n name of the object to open
        */
        object open(const String &n) const
        {
            typedef typename NXObjectMap<GType>::ObjectType ObjectType;
            typedef typename NXObjectMap<GType>::GroupType GroupType;
            typedef typename NXObjectMap<GType>::FieldType FieldType;
           
            //open the NXObject 
            ObjectType nxobject = this->_object.open(n);

            if(nxobject.object_type() == pni::nx::NXObjectType::NXFIELD)
            {
                NXFieldWrapper<FieldType> *ptr = new
                    NXFieldWrapper<FieldType>(FieldType(nxobject));
                return object(ptr);
            }

            if(nxobject.object_type() == pni::nx::NXObjectType::NXGROUP)
            {
                //we use here copy construction thus we do not have to care
                //of the original nxobject goes out of scope and gets destroyed.
                NXGroupWrapper<GroupType> *ptr = new
                    NXGroupWrapper<GroupType>(GroupType(nxobject));
                return object(ptr);
            }

            //should raise an exception here

            //this here is to avoid compiler warnings
            return object();

        }

        //--------------------------------------------------------------------------
        //! wrap the exists method
        bool exists(const String &n) const
        {
            return this->_object.exists(n);
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


};


template<typename GType> void wrap_nxgroup(const String &class_name)
{

    class_<NXGroupWrapper<GType>,bases<NXObjectWrapper<GType> > >(class_name.c_str())
        .def(init<>())
        .def("open",&NXGroupWrapper<GType>::open)
        .def("create_group",&NXGroupWrapper<GType>::create_group)
        .def("create_field",&NXGroupWrapper<GType>::create_scalar_field)
        .def("create_field",&NXGroupWrapper<GType>::create_array_field)
        .def("create_field",&NXGroupWrapper<GType>::create_array_field_chunked)
        .def("exists",&NXGroupWrapper<GType>::exists)
        .def("link",&NXGroupWrapper<GType>::link)
        ;
}


#endif
