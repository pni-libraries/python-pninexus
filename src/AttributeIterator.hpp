/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of libpninx-python.
 *
 * libpninx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * libpninx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libpninx.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Iterator for childs linked to a group or file object
 *
 * Created on: March 14, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __ATTRIBUTEITERATOR_HPP__
#define __ATTRIBUTEITERATOR_HPP__

//! \brief exception to stop iteration
class AttributeIteratorStop:public std::exception
{

};


//! \brief child iterator

//! This is a forward iterator that runs through all the objects
//! linked below a group. 
template<typename IterableT,typename ItemT> class AttributeIterator
{
    private:
        const IterableT *_parent; //!< parent object of the interator
        size_t     _nattrs; //!< total number of links
        size_t     _index;  //!< actual index 
        ItemT      _item;   //!< the actual object to which the 
                            //!< interator referes
    public:
        typedef ItemT&    reference;
        typedef ItemT     value_type;
        typedef IterableT iterable_type;
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
        //! constructor from group object
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
        void increment(){
            _index++;
            if(_index < _nattrs){
                _item = _parent->open_attr_by_id(_index);
            }
        }

        //---------------------------------------------------------------------
        ItemT next()
        {
            //check if iteration is still possible
            if(_index == _nattrs){
                //raise exception here
                throw(AttributeIteratorStop());
                return(ItemT());
            }

            ItemT item(_item);
            this->increment();

            return item;
        }

        //----------------------------------------------------------------------
        object __iter__()
        {
            return object(this);
        }

};

template<typename Iterable,typename ItemT> void wrap_attributeiterator(const String &class_name)
{
    class_<AttributeIterator<Iterable,ItemT> >(class_name.c_str())
        .def(init<>())
        .def("next",&AttributeIterator<Iterable,ItemT>::next)
        .def("__iter__",&AttributeIterator<Iterable,ItemT>::__iter__)
        ;
}

#endif
