/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of python-pniio.
 *
 * python-pniio is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * python-pniio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Class-template for field creators.
 *
 * Created on: Feb 17, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */
#pragma once

#include <pni/io/nx/NX.hpp>

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
template<> class NXObjectMap<pni::io::nx::h5::NXObject>
{
    public:
        typedef pni::io::nx::h5::NXObject ObjectType;
        typedef pni::io::nx::h5::NXGroup GroupType;
        typedef pni::io::nx::h5::NXField FieldType;
        typedef pni::io::nx::h5::NXAttribute AttributeType;
};

template<> class NXObjectMap<pni::io::nx::h5::NXGroup>
{
    public:
        typedef pni::io::nx::h5::NXObject ObjectType;
        typedef pni::io::nx::h5::NXGroup GroupType;
        typedef pni::io::nx::h5::NXField FieldType;
        typedef pni::io::nx::h5::NXAttribute AttributeType;
};

template<> class NXObjectMap<pni::io::nx::h5::NXField>
{
    public:
        typedef pni::io::nx::h5::NXObject ObjectType;
        typedef pni::io::nx::h5::NXGroup GroupType;
        typedef pni::io::nx::h5::NXField FieldType;
        typedef pni::io::nx::h5::NXAttribute AttributeType;
};

template<> class NXObjectMap<pni::io::nx::h5::NXFile>
{
    public:
        typedef pni::io::nx::h5::NXObject ObjectType;
        typedef pni::io::nx::h5::NXGroup GroupType;
        typedef pni::io::nx::h5::NXField FieldType;
        typedef pni::io::nx::h5::NXAttribute AttributeType;
};

//! \endcond
