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
 * Created on: March 13, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __CHILDITERATOR_HPP__
#define __CHILDITERATOR_HPP__

//! \brief exception to stop iteration
class ChildIteratorStop:public std::exception
{

};


//! \brief child iterator

//! This is a forward iterator that runs through all the objects
//! linked below a group. 
template<typename IterableT,typename ItemT> class ChildIterator
{
    private:
        const IterableT *_parent; //!< parent object of the interator
        size_t     _nlinks; //!< total number of links
        size_t     _index;  //!< actual index 
        ItemT      _item;   //!< the actual object to which the 
                            //!< interator referes
    public:
        typedef ItemT&    reference;
        typedef ItemT     value_type;
        typedef IterableT iterable_type;
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


        void increment(){
            _index++;
            if(_index < _nlinks){
                _item = _parent->open(_index);
            }
        }

        //---------------------------------------------------------------------
        ItemT next()
        {
            //check if iteration is still possible
            if(_index == _nlinks){
                //raise exception here
                throw(ChildIteratorStop());
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

template<typename Iterable> void wrap_childiterator(const String &class_name)
{
    class_<ChildIterator<Iterable,object> >(class_name.c_str())
        .def(init<>())
        .def("next",&ChildIterator<Iterable,object>::next)
        .def("__iter__",&ChildIterator<Iterable,object>::__iter__)
        ;
}

#endif
