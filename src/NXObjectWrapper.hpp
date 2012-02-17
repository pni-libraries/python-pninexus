#ifndef __NXOBJECTWRAPPER_HPP__
#define __NXOBJECTWRAPPER_HPP__

#include <pni/utils/Types.hpp>


template<typename OType> class NXObjectWrapper
{
    protected:
        OType _object;
    public:
        //================constructors and destructor==========================
        //! default constructor
        NXObjectWrapper():_object(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXObjectWrapper(const NXObjectWrapper<OType> &o):
            _object(o._object)
        {
        }

        //---------------------------------------------------------------------
        //! move constructor
        NXObjectWrapper(NXObjectWrapper<OType> &&o):
            _object(std::move(o._object)) 
        {
            if(_object.is_valid()){
                std::cerr<<"everything went fine!"<<std::endl;
            }else{
                std::cerr<<"something went wrong!"<<std::endl;
            }
        }

        //---------------------------------------------------------------------
        //! copy conversion constructor from wrapped object
        explicit NXObjectWrapper(const OType &o):_object(o){}

        //---------------------------------------------------------------------
        //! move conversion constructor from wrapped object
        explicit NXObjectWrapper(OType &&o):_object(std::move(o)){
        }

        //==================assignment operators===============================
        //! copy conversion assignment from wrapped type
        NXObjectWrapper<OType> &operator=(const OType &o)
        {
            _object = o;
            return *this;
        }

        //---------------------------------------------------------------------
        //! move conversion assignment from wrapped type
        NXObjectWrapper<OType> &operator=(OType &&o)
        {
            _object = std::move(o);
            return *this;
        }

        //---------------------------------------------------------------------
        //move assignment
        NXObjectWrapper<OType> &operator=(NXObjectWrapper<OType> &&o)
        {
            if(this != &o) _object = std::move(o._object);
            return *this;
        }

        //---------------------------------------------------------------------
        //copy assignment
        NXObjectWrapper<OType> &operator=(const NXObjectWrapper<OType> &o)
        {
            if(this != &o) _object = o._object;
            return *this;
        }


        //======================object methods=================================

        //! obtain base name
        String base() const
        {
            return _object.base();
        }

        //! obtain name
        String name() const
        {
            return _object.name();
        }

        //! obtain path
        String path() const
        {
            return _object.path();
        }

        //! get validity status
        bool is_valid() const
        {
            return _object.is_valid();
        }


        NXAttribute attr(const String &name,TypeID &id){

        }

        NXAttribute attr(const String &name,TypeID &id,const Shape &s){

        }
};

//template function wrapping a single NXObject 
//type. 




template<typename OType> void wrap_nxobject(const String &class_name)
{
    class_<NXObjectWrapper<OType> >(class_name.c_str())
        .def(init<>())
        .add_property("name",&NXObjectWrapper<OType>::name)
        .add_property("path",&NXObjectWrapper<OType>::path)
        .add_property("base",&NXObjectWrapper<OType>::base)
        .add_property("is_valid",&NXObjectWrapper<OType>::is_valid)
        ;
}



#endif
