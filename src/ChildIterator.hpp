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
 * Iterator for childs linked to a group or file object
 *
 * Created on: March 13, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#pragma once

#include "NXWrapperErrors.hpp"

/*! 
\ingroup iterators  
\brief child iterator

This is a forward iterator that runs through all the objects
linked below a group. 
*/
template<typename IterableT,typename ItemT> class ChildIterator
{
    private:
        const IterableT *_parent; //!< parent object of the interator
        size_t     _nlinks; //!< total number of links
        size_t     _index;  //!< actual index 
        ItemT      _item;   //!< the actual object to which the 
                            //!< interator referes
    public:
        typedef ItemT     value_type;    //!< type of the elements
        typedef IterableT iterable_type; //!< container type
        //=======================constructors and destructor====================
        //! default constructor
        ChildIterator():
            _parent(nullptr),
            _nlinks(0),
            _index(0),
            _item()
        {}   
        
        //---------------------------------------------------------------------
        //! copy constructor
        ChildIterator(const ChildIterator<IterableT,ItemT> &i):
            _parent(i._parent),
            _nlinks(i._nlinks),
            _index(i._index),
            _item(i._item)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        ChildIterator(ChildIterator<IterableT,ItemT> &&i):
            _parent(i._parent),
            _nlinks(i._nlinks),
            _index(i._index),
            _item(std::move(i._item))
        {
            i._parent = nullptr;
            i._nlinks = 0;
            i._index  = 0;
        }

        //---------------------------------------------------------------------
        //! constructor from group object
        ChildIterator(const IterableT &g,size_t start_index=0):
            _parent(&g),
            _nlinks(g.nchilds()),
            _index(start_index),
            _item()
        {
            if(_index < _nlinks) _item = _parent->open(_index);
        }

        //---------------------------------------------------------------------
        //! destructor
        virtual ~ChildIterator(){
            _parent = nullptr;
            _nlinks = 0;
            _index  = 0;
        }

        //=======================assignment operators==========================
        //! copy assignment operator
        ChildIterator<IterableT,ItemT> &
            operator=(const ChildIterator<IterableT,ItemT> &i)
        {
            if(this != &i){
                _parent = i._parent;
                _nlinks = i._nlinks;
                _index  = i._index;
                _item   = i._item;
            }
            return *this;
        }

        //---------------------------------------------------------------------
        //! move assignment operator
        ChildIterator<IterableT,ItemT> &
            operator=(ChildIterator<IterableT,ItemT> &&i)
        {
            if(this != &i){
                _parent = i._parent;
                i._parent = nullptr;
                _nlinks = i._nlinks;
                i._nlinks = 0;
                _index  = i._index;
                i._index = 0;
                _item   = std::move(i._item);
            }
            return *this;
        }


        //---------------------------------------------------------------------
        /*! \brief increment iterator

        Set the iterator to the next element in the container.
        */
        void increment()
        {
            _index++;
            if(_index < _nlinks)
                _item = _parent->open(_index);
        }

        //---------------------------------------------------------------------
        /*! \brief return current element

        Return the current element the iterator points to.
        \return intance of ItemT
        */
        ItemT next()
        {
            //check if iteration is still possible
            if(_index == _nlinks)
            {
                //raise exception here
                throw(ChildIteratorStop());
                return(ItemT());
            }

            ItemT item(_item);
            this->increment();

            return item;
        }

        //----------------------------------------------------------------------
        //! \brief required by the python wrapper
        object __iter__()
        {
            return object(this);
        }

};

/*! \brief creates Python object

Function creates a Python object for a ChildIterator. The type of the container
is determined by the Iterable template parameter.
\param class_name name of the created class
*/
template<typename Iterable> void wrap_childiterator(const string &class_name)
{
    class_<ChildIterator<Iterable,object> >(class_name.c_str())
        .def(init<>())
        .def("next",&ChildIterator<Iterable,object>::next)
        .def("__iter__",&ChildIterator<Iterable,object>::__iter__)
        ;
}
