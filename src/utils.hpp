//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pnicore.
//
// python-pnicore is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pnicore is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pnicore.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Oct 21, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

extern "C"{
#include<Python.h>
#define NO_IMPORT_ARRAY
#include<numpy/arrayobject.h>
}

#include <vector>
#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>


#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;


//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief create a python list from a container
//! 
//! Creates a Python list from a C++ container.
//! \tparam CTYPE containerr type
//! \param c instance of CTYPE
//! \return python list with 
//!
template<typename CTYPE> list Container2List(const CTYPE &c)
{
    list l;
    if(c.size()==0) return l;

    for(auto iter=c.begin();iter!=c.end();++iter)
        l.append(*iter);

    return l;

}

//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief create a container from a Python list
//!
//! Convert a Python list to a C++ container. 
//! \tparam CTYPE container type.
//! \param l python list object
//! \return instance of 
//!
template<typename CTYPE> CTYPE List2Container(const list &l)
{
    //if the list is empty we return an empty container
    if(!len(l)) return CTYPE();

    //otherwise we need to copy some content
    CTYPE c(len(l));

    size_t index=0;
    for(typename CTYPE::iterator iter=c.begin();iter!=c.end();++iter)
        *iter = extract<typename CTYPE::value_type>(l[index++]);

    return c;
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief tuple to container conversion
//! 
//! Converts a Python tuple to a Shape object. The length of the tuple 
//! determines the rank of the Shape and its elements the number of elements 
//! along each dimension.
//! 
//! \tparam CTYPE container type
//! \param t tuple object
//! \return instance of type CTYPE
//!
template<typename CTYPE> CTYPE Tuple2Container(const tuple &t)
{
    return List2Container<CTYPE>(list(t));
}

//-----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief check if unicode
//! 
//! Check if the instance of objec represents a unicode object. 
//! \param o instance to check
//! \return true if o is a unicode object, false otherwise
//!
bool is_unicode(const object &o);

//----------------------------------------------------------------------------
bool is_int(const object &o);

//----------------------------------------------------------------------------
bool is_bool(const object &o);

//----------------------------------------------------------------------------
bool is_long(const object &o);

//----------------------------------------------------------------------------
bool is_float(const object &o);

//----------------------------------------------------------------------------
bool is_complex(const object &o);

//----------------------------------------------------------------------------
bool is_string(const object &o);

//----------------------------------------------------------------------------
bool is_scalar(const object &o);

//-----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief convert unicode to string
//! 
//! Converts a Python unicode object to a common Python String object using 
//! UTF8 encoding.
//! 
//! \param o python unicode object
//! \return python string object
//!
object unicode2str(const object &o);

