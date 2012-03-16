#ifndef __NXOBJECTMAP_HPP__
#define __NXOBJECTMAP_HPP__

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
        typedef void SelectionType; //!< wrapped selection type
        typedef void AttributeType; //!< wrapped attribute type
};

#endif
