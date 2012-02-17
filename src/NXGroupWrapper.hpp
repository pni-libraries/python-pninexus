#ifndef __NXGROUPWRAPPER_HPP__
#define __NXGROUPWRAPPER_HPP__

#include "NXWrapperHelpers.hpp"
#include "NXObjectMap.hpp"
#include "NXObjectWrapper.hpp"


template<typename GType> class NXGroupWrapper:public NXObjectWrapper<GType>
{
    public:
        //================constructors and destructor==========================
        //! default constructor
        NXGroupWrapper():NXObjectWrapper<GType>(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXGroupWrapper(const NXGroupWrapper<GType> &o):
            NXObjectWrapper<GType>(o){}

        //----------------------------------------------------------------------
        //! move constructor
        NXGroupWrapper(NXGroupWrapper<GType> &&o):
            NXObjectWrapper<GType>(std::move(o))
        {
        }

        //----------------------------------------------------------------------
        //! conversion copy constructor
        explicit NXGroupWrapper(const GType &g):NXObjectWrapper<GType>(g){}

        //----------------------------------------------------------------------
        //! conversion move constructor
        explicit NXGroupWrapper(GType &&g):NXObjectWrapper<GType>(std::move(g)){}


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

        //! create a group
        NXGroupWrapper<typename NXObjectMap<GType>::GroupType > create_group(const String &n) const
        {
            typedef typename NXObjectMap<GType>::GroupType GroupType;
            return NXGroupWrapper<GroupType> (this->_object.create_group(n));
        }

        //! open a type
        NXObjectWrapper<typename NXObjectMap<GType>::ObjectType > 
            open(const String &n) const
        {
            typedef typename NXObjectMap<GType>::ObjectType ObjectType;
            return NXObjectWrapper<ObjectType>(this->_object.open(n));
        }
};


template<typename GType> void wrap_nxgroup(const String &class_name)
{
    class_<NXGroupWrapper<GType>,bases<NXObjectWrapper<GType> > >(class_name.c_str())
        .def(init<>())
        .def("open",&NXGroupWrapper<GType>::open)
        .def("create_group",&NXGroupWrapper<GType>::create_group)
        ;
}

template<typename GType> void wrap_nxgroup_nocop(const String &class_name)
{
    class_<NXGroupWrapper<GType>,bases<NXObjectWrapper<GType>
        >,boost::noncopyable >(class_name.c_str())
        .def(init<>())
        .def("open",&NXGroupWrapper<GType>::open)
        .def("create_group",&NXGroupWrapper<GType>::create_group)
        ;
}

#endif
