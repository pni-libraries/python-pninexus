#ifndef __NXGROUPWRAPPER_HPP__
#define __NXGROUPWRAPPER_HPP__

#include <pni/nx/NXObjectType.hpp>

#include "NXWrapperHelpers.hpp"
#include "NXObjectMap.hpp"
#include "NXObjectWrapper.hpp"
#include "NXFieldWrapper.hpp"
#include "FieldCreator.hpp"
#include "ChildIterator.hpp"
#include "AttributeIterator.hpp"


template<typename GType> class NXGroupWrapper:public NXObjectWrapper<GType>
{
    public:
        typedef NXFieldWrapper<typename NXObjectMap<GType>::FieldType> field_type;
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

        //----------------------------------------------------------------------
        //! create group with NXclass
        NXGroupWrapper<typename NXObjectMap<GType>::GroupType >
            create_group(const String &n,const String &nxclass=String()) const
        {
            typedef typename NXObjectMap<GType>::GroupType GroupType;
            NXGroupWrapper<GroupType> g(this->_object.create_group(n,nxclass));
            return g;
        }

        //-------------------------------------------------------------------------
        //! \brief create field without filter

        field_type create_field_nofilter(const String &name,const String
                &type_code,const object &shape=list(),const object
                &chunk=list(),const object &filter=object())
        
        {
            FieldCreator<field_type> creator(name,
                    List2Shape(list(shape)),List2Shape(list(chunk)),filter);
            return creator.create(this->_object,type_code);
        }

        //-------------------------------------------------------------------------
        /*! \brief open an object

        This method opens an object and tries to figure out by itself what kind
        of object is has to deal with. Consequently it returns already a Python
        obeject of the appropriate type. 
        \param n name of the object to open
        */
        object open_by_name(const String &n) const
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

        //---------------------------------------------------------------------
        /*! \brief open object by index

        Opens a child of the group by its index.
        \param i index of the child
        \return child object
        */
        object open(size_t i) const
        {
            typedef typename NXObjectMap<GType>::ObjectType ObjectType;
            ObjectType nxobject = this->_object.open(i);

            return open_by_name(nxobject.path());
        }

        //--------------------------------------------------------------------------
        //! wrap the exists method
        bool exists(const String &n) const
        {
            return this->_object.exists(n);
        }

        //--------------------------------------------------------------------------
        size_t nchilds() const
        {
            return this->_object.nchilds();
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

        //----------------------------------------------------------------------
        ChildIterator<NXGroupWrapper<GType>,object> get_child_iterator() const
        {
            return ChildIterator<NXGroupWrapper<GType>,object>(*this);
        }

};


template<typename GType> void wrap_nxgroup(const String &class_name)
{

    class_<NXGroupWrapper<GType>,bases<NXObjectWrapper<GType> > >(class_name.c_str())
        .def(init<>())
        .def("open",&NXGroupWrapper<GType>::open_by_name)
        .def("__getitem__",&NXGroupWrapper<GType>::open)
        .def("__getitem__",&NXGroupWrapper<GType>::open_by_name)
        .def("create_group",&NXGroupWrapper<GType>::create_group,("n",arg("nxclass")=String()))
        .def("create_field",&NXGroupWrapper<GType>::create_field_nofilter,
                ("name","type_code",arg("shape")=list(),arg("chunk")=list(),arg("filter")=object()))
        .def("exists",&NXGroupWrapper<GType>::exists)
        .def("link",&NXGroupWrapper<GType>::link)
        .add_property("nchilds",&NXGroupWrapper<GType>::nchilds)   
        .add_property("childs",&NXGroupWrapper<GType>::get_child_iterator)
        ;
}


#endif
