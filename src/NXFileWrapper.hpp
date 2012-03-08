#ifndef __NXFILEWRAPPER_HPP__
#define __NXFILEWRAPPER_HPP__

#include "NXObjectWrapper.hpp"
#include "NXGroupWrapper.hpp"

template<typename FType> class NXFileWrapper:public NXGroupWrapper<FType>
{
    public:
        //==================constructor and destructor=========================
        //! default constructor
        NXFileWrapper():NXGroupWrapper<FType>(){}

        //----------------------------------------------------------------------
        //! copy constructor
        NXFileWrapper(const NXFileWrapper<FType> &o):
            NXGroupWrapper<FType>(o){}

        //----------------------------------------------------------------------
        //! move constructor
        NXFileWrapper(NXFileWrapper<FType> &&f):
            NXGroupWrapper<FType>(std::move(f)){}

        //-----------------------------------------------------------------------
        //! move conversion constructor from wrapped object
        explicit NXFileWrapper(FType &&f):NXGroupWrapper<FType>(std::move(f)){}

        //-----------------------------------------------------------------------
        //! copy conversion constructor from wrapped object
        explicit NXFileWrapper(const FType &f):NXGroupWrapper<FType>(f){}

        //----------------------------------------------------------------------
        //! destructor
        ~NXFileWrapper()
        {
        }

        //=======================assignment operators===========================
        //! move conversion assignment from wrapped object
        NXFileWrapper<FType> &operator=(FType &&f)
        {
            NXGroupWrapper<FType>::operator=(f);
            return *this;
        }

        //-----------------------------------------------------------------------
        //! copy conversion assignment from wrapped object
        NXFileWrapper<FType> &operator=(const FType &f)
        {
            NXGroupWrapper<FType>::operator=(f);
            return *this;
        }

        //------------------------------------------------------------------------
        //! copy assignment
        NXFileWrapper<FType> &operator=(const NXFileWrapper<FType> &f)
        {
            if(this != &f) NXGroupWrapper<FType>::operator=(f);
            return *this;
        }

        //-------------------------------------------------------------------------
        //! move assignment
        NXFileWrapper<FType> &operator=(NXFileWrapper<FType> &&f)
        {
            if(this != &f) NXGroupWrapper<FType>::operator=(std::move(f));
            return *this;
        }
};

//-----------------------------------------------------------------------------
template<typename FType> NXFileWrapper<FType> create_file(const String &n,
        bool ov=true,ssize_t s=0)
{
    NXFileWrapper<FType> file(FType::create_file(n,ov,s));
    return file;
}

//------------------------------------------------------------------------------
template<typename FType> NXFileWrapper<FType> open_file(const String &n,
        bool ro=false)
{
    return NXFileWrapper<FType>(FType::open_file(n,ro)); 
}

//------------------------------------------------------------------------------
template<typename FType> void wrap_nxfile(const String &class_name)
{
    
    class_<NXFileWrapper<FType>,bases<NXGroupWrapper<FType> > >(class_name.c_str())
        .def(init<>())
        ;

    //need some functions
    def("create_file",&create_file<FType>);
    def("open_file",&open_file<FType>);
}

#endif
