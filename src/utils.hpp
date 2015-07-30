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


#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>

//-----------------------------------------------------------------------------
//! 
//! \ingroup pub_api  
//! \brief create a python list from a C++ container
//!
//! Takes an arbitrary C++ container which must support the STL iterator 
//! protocoll and copy its content to a Python list. 
//! 
//! \tparam CTYPE C++ container type
//! \param c instance of CTYPE
//! \return python list with 
//!
template<typename CTYPE> boost::python::list Container2List(const CTYPE &c)
{
    boost::python::list l;
    if(c.size()==0) return l;

    for(auto iter=c.begin();iter!=c.end();++iter)
        l.append(*iter);

    return l;

}

//-----------------------------------------------------------------------------
//! 
//! \ingroup pub_api  
//! \brief create a C++ container from a Python list
//!
//! Copies the content of a Python list to an arbitrary C++ container. 
//! This container must provided the standard STL iterator interface and 
//! must be constructable from its size. The elements of the list must be 
//! convertible to the value_type of the container.
//! 
//! \tparam CTYPE C++ container type.
//! \param l python list object
//! \return instance of 
//!
template<typename CTYPE> CTYPE List2Container(const boost::python::list &l)
{
    //if the list is empty we return an empty container
    if(!boost::python::len(l)) return CTYPE();

    //otherwise we need to copy some content
    CTYPE c(len(l));

    size_t index=0;
    for(typename CTYPE::iterator iter=c.begin();iter!=c.end();++iter)
        *iter = boost::python::extract<typename CTYPE::value_type>(l[index++]);

    return c;
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup pub_api  
//! \brief copy a tuple to a C++ container 
//! 
//! Copy the content of a Python tuple to a C++ container. Basically, this 
//! function converts the tuple to a Python list and calls 
//! List2Container. Thus, the container type must be construtable from its 
//! size and the elements of the tuple must be convertible to the 
//! value_type of the container. 
//! 
//! \tparam CTYPE C++ container type
//! \param t tuple object
//! \return instance of type CTYPE
//!
template<typename CTYPE> CTYPE Tuple2Container(const boost::python::tuple &t)
{
    return List2Container<CTYPE>(boost::python::list(t));
}

//-----------------------------------------------------------------------------
//!
//! \ingroup pub_api
//! \brief check if unicode
//! 
//! Check if the instance of object represents a unicode object. For Python 2
//! this must be a specially created object, for Python 3 every default 
//! constructed string is a unicode object.
//! 
//! \param o instance to check
//! \return true if o is a unicode object, false otherwise
//!
bool is_unicode(const boost::python::object &o);

//----------------------------------------------------------------------------
//! 
//! \ingroup pub_api
//! \brief check if object is an integer
//! 
//! Returns true if the passed Python object is an integer. For Python 3 
//! this function is equivalent to is_long. 
//! 
//! \param o reference to a Python object
//! \return true if integer, false otherwise 
//! 
bool is_int(const boost::python::object &o);

//----------------------------------------------------------------------------
//! 
//! \ingroup pub_api
//! \brief check if object is a boolean 
//! 
//! \param o reference to Python object
//! \return true if object is a boolean value, false otherwise
//! 
bool is_bool(const boost::python::object &o);

//----------------------------------------------------------------------------
//!
//! \ingroup pub_api
//! \brief check if object is long 
//! 
//! \param o reference to Python object
//! \return true if o is an instance of long, false otherwise
//!
bool is_long(const boost::python::object &o);

//----------------------------------------------------------------------------
//! 
//! \ingroup pub_api
//! \brief check if object is float
//!
//! \param o reference to Python object
//! \return true if o is a float instance, false otherwise
//! 
bool is_float(const boost::python::object &o);

//----------------------------------------------------------------------------
//!
//! \ingroup pub_api
//! \brief check if object is complex
//! 
//! \param o reference to Python object
//! \return true if o is an instance of complex, false otherwise
//!
bool is_complex(const boost::python::object &o);

//----------------------------------------------------------------------------
//! 
//! \ingroup pub_api
//! \brief check if object is a string
//!
//! \todo we need to check the behavior of this function on Python2 and 
//! Python 3
//! 
//! \param o reference to Python object
//! \return true if o is an instance of string, false otherwise 
//! 
bool is_string(const boost::python::object &o);

//----------------------------------------------------------------------------
//!
//! \ingroup pub_api
//! \brief check if the object is a scalar 
//! 
//! This function returns true if the object is a scalar value. As scalars
//! we currently consider the following types 
//! 
//! - `unicode` and `string` instances
//! 
//! - the numeric types `int`, `float`, `long`, and `complex`
//! 
//! - numpy scalars (basically arrays with a single element)
//! 
//! \param o reference to the Python object
//! \return true if o is one of the above types, false otherwise
//! 
bool is_scalar(const boost::python::object &o);

//-----------------------------------------------------------------------------
//!
//! \ingroup pub_api
//! \brief convert unicode to string
//! 
//! Converts a Python unicode object to a common Python String object using 
//! UTF8 encoding.
//!
//! \todo we need to check if this function is useful on Python3 at all
//! 
//! \param o python unicode object
//! \return python string object
//!
boost::python::object unicode2str(const boost::python::object &o);

