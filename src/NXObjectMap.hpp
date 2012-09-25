#ifndef __NXOBJECTMAP_HPP__
#define __NXOBJECTMAP_HPP__

#include <pni/nx/NX.hpp>

/*! 
\ingroup utils  
\brief object map for wrapped types

This map provides a type-map from a particular implementation type to all other
related types. So for instance if you wrap a particular NXGroup object this
type-map will provide all the other types related to this particular type.
This is particularly useful in cases where you need to determine for instance
the AttributeType from a group type.
*/
template<typename GType> class NXObjectMap{
    public:
        typedef void ObjectType;    //!< wrapped object type
        typedef void GroupType;     //!< wrapped group type
        typedef void FieldType;     //!< wrapped field type
        typedef void AttributeType; //!< wrapped attribute type
};


//! \cond NO_API_DOC
template<> class NXObjectMap<pni::nx::h5::NXObject>
{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXAttribute AttributeType;
};

template<> class NXObjectMap<pni::nx::h5::NXGroup>
{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXAttribute AttributeType;
};

template<> class NXObjectMap<pni::nx::h5::NXField>
{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXAttribute AttributeType;
};

template<> class NXObjectMap<pni::nx::h5::NXFile>
{
    public:
        typedef pni::nx::h5::NXObject ObjectType;
        typedef pni::nx::h5::NXGroup GroupType;
        typedef pni::nx::h5::NXField FieldType;
        typedef pni::nx::h5::NXAttribute AttributeType;
};

//! \endcond
#endif
