#ifndef __NXGROUPWRAPPER_HPP__
#define __NXGROUPWRAPPER_HPP__

#include <pni/nx/NXObjectType.hpp>

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
                return object();
            }

            if(nxobject.object_type() == pni::nx::NXObjectType::NXGROUP)
            {
                //we use here copy construction thus we do not have to care
                //of the original nxobject goes out of scope and gets destroyed.
                NXGroupWrapper<GroupType> *ptr = new
                    NXGroupWrapper<GroupType>(GroupType(nxobject));
                return object(ptr);
            }

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
        .def("exists",&NXGroupWrapper<GType>::exists)
        .def("link",&NXGroupWrapper<GType>::link)
        ;
}

template<typename GType> void wrap_nxgroup_nocop(const String &class_name)
{
    class_<NXGroupWrapper<GType>,bases<NXObjectWrapper<GType>
        >,boost::noncopyable >(class_name.c_str())
        .def(init<>())
        .def("open",&NXGroupWrapper<GType>::open)
        .def("create_group",&NXGroupWrapper<GType>::create_group)
        .def("exists",&NXGroupWrapper<GType>::exists)
        .def("link",&NXGroupWrapper<GType>::link)
        ;
}

#endif
