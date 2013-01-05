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
 * Iterator for attributes attached to an  object.
 *
 * Created on: March 14, 2012
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
template<typename IterableT,typename ItemT> class AttributeIterator
{
    private:
        const IterableT *_parent; //!< parent object of the interator
        size_t     _nattrs; //!< total number of links
        size_t     _index;  //!< actual index 
        ItemT      _item;   //!< the actual object to which the 
                            //!< interator referes
    public:
        typedef ItemT     value_type;    //!< type of the elements 
        typedef IterableT iterable_type; //!< type of the iterable
        //=======================constructors and destructor====================
        //! default constructor
        AttributeIterator():
            _parent(nullptr),
            _nattrs(0),
            _index(0),
            _item()
        {}   
        
        //---------------------------------------------------------------------
        //! copy constructor
        AttributeIterator(const AttributeIterator<IterableT,ItemT> &i):
            _parent(i._parent),
            _nattrs(i._nattrs),
            _index(i._index),
            _item(i._item)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        AttributeIterator(AttributeIterator<IterableT,ItemT> &&i):
            _parent(i._parent),
            _nattrs(i._nattrs),
            _index(i._index),
            _item(std::move(i._item))
        {
            i._parent = nullptr;
            i._nattrs = 0;
            i._index  = 0;
        }

        //---------------------------------------------------------------------
        /*! \brief constructor from group object

        \param g iterable object 
        \param start_index index of the first element the iterator should point
        to
        */
        explicit AttributeIterator(const IterableT &g,size_t start_index=0):
            _parent(&g),
            _nattrs(g.nattrs()),
            _index(start_index),
            _item()
        {
            if(_index < _nattrs)
                _item = _parent->open_attr_by_id(_index);
        }

        //---------------------------------------------------------------------
        //! destructor
        virtual ~AttributeIterator(){
            _parent = nullptr;
            _nattrs = 0;
            _index  = 0;
        }

        //=======================assignment operators==========================
        //! copy assignment operator
        AttributeIterator<IterableT,ItemT> &
            operator=(const AttributeIterator<IterableT,ItemT> &i)
        {
            if(this != &i){
                _parent = i._parent;
                _nattrs = i._nattrs;
                _index  = i._index;
                _item   = i._item;
            }
            return *this;
        }

        //---------------------------------------------------------------------
        //! move assignment operator
        AttributeIterator<IterableT,ItemT> &
            operator=(AttributeIterator<IterableT,ItemT> &&i)
        {
            if(this != &i){
                _parent = i._parent;
                i._parent = nullptr;
                _nattrs = i._nattrs;
                i._nlinks = 0;
                _index  = i._index;
                i._index = 0;
                _item   = std::move(i._item);
            }
            return *this;
        }

        //---------------------------------------------------------------------
        /*! \brief increment iterator

        Moves the iterator to the next element.
        */
        void increment(){
            _index++;
            if(_index < _nattrs){
                _item = _parent->open_attr_by_id(_index);
            }
        }

        //---------------------------------------------------------------------
        /*! \brief return next element

        This method returns the next element of the container. 
        \return instance of ItemT with the next element
        */
        ItemT next()
        {
            //check if iteration is still possible
            if(_index == _nattrs){
                //raise exception here
                throw(AttributeIteratorStop());
                return(ItemT());
            }

            //return the current object 
            ItemT item(_item);
            //increment the iterator
            this->increment();

            return item;
        }

        //----------------------------------------------------------------------
        //! \brief required by Python
        object __iter__()
        {
            return object(this);
        }

};

//-----------------------------------------------------------------------------
/*! \brief AttributeIterator wrapper generator

This function creates the Python code for AttributeIterator objects. 
\param class_name name of the created Python class
*/
template<typename Iterable,typename ItemT> 
void wrap_attributeiterator(const String &class_name)
{
    class_<AttributeIterator<Iterable,ItemT> >(class_name.c_str())
        .def(init<>())
        .def("next",&AttributeIterator<Iterable,ItemT>::next)
        .def("__iter__",&AttributeIterator<Iterable,ItemT>::__iter__)
        ;
}

