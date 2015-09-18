//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pniio.
//
// python-pniio is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pniio is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: March 13, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#pragma once

#include <boost/python.hpp>
#include <pni/core/types.hpp>
#include "errors.hpp"

//! 
//! \ingroup iterators  
//! \brief child iterator
//! 
//! This is a forward iterator that runs through all the objects
//! linked below a group. 
//!
//! \tparam CT type of the iterable 
//!
template<typename CT> class child_iterator
{
    public:
        typedef typename CT::object_type     value_type;    //!< type of the elements
        typedef CT  iterable_type; //!< container type
    private:
        const CT *_parent; //!< parent object of the interator
        size_t     _nlinks; //!< total number of links
        size_t     _index;  //!< actual index 
        value_type     _item;   //!< the actual object to which the 
                            //!< interator referes
    public:
        //=======================constructors and destructor====================
        //! default constructor
        child_iterator():
            _parent(nullptr),
            _nlinks(0),
            _index(0),
            _item()
        {}   
        
        //---------------------------------------------------------------------
        //! constructor from group object
        explicit child_iterator(const iterable_type &g,size_t start_index=0):
            _parent(&g),
            _nlinks(g.nchildren()),
            _index(start_index),
            _item()
        {
            if(_index < _nlinks) _item = _parent->open_by_index(_index);
        }

        //---------------------------------------------------------------------
        //! 
        //! \brief increment iterator
        //! 
        //! Set the iterator to the next element in the container.
        //!
        void increment()
        {
            _index++;
            if(_index < _nlinks)
                _item = _parent->open_by_index(_index);
        }

        //---------------------------------------------------------------------
        //! 
        //! \brief return current element
        //!
        //! Return the current element the iterator points to.
        //! \return intance of ItemT
        //!
        boost::python::object next()
        {
            using namespace boost::python;

            //check if iteration is still possible
            if(_index >= _nlinks)
            {
                //raise exception here
                throw(ChildIteratorStop());
                return(object());
            }

            object item(_item);
            this->increment();

            return item;
        }

        //----------------------------------------------------------------------
        //! \brief required by the python wrapper
        boost::python::object __iter__()
        {
            return boost::python::object(this);
        }

};

//! 
//! \brief creates Python object
//! 
//! Function creates a Python object for a ChildIterator. The type of the 
//! container is determined by the Iterable template parameter.
//!
template<typename CT> void wrap_childiterator(const pni::core::string &class_name)
{
    using namespace boost::python;

    typedef child_iterator<CT> iterator_type;

    class_<iterator_type>(class_name.c_str())
        .def(init<>())
        .def("next",&iterator_type::next)
        .def("__iter__",&iterator_type::__iter__)
        ;
}
