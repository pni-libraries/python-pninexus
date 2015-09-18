//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// along with pyton-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Sep 16, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#pragma once

#include <boost/python.hpp>
#include <pni/io/nx/flat_group.hpp>

template<typename GTYPE> 
class rec_group_iterator
{
    public:
        typedef pni::io::nx::flat_group<GTYPE> group_type;
        typedef std::shared_ptr<group_type> group_ptr;
        typedef rec_group_iterator<GTYPE> iterator_type;
    private:
        group_ptr _flat_group; 
        size_t _index;
    public:
        rec_group_iterator():_flat_group(),_index(0) 
        {}

        rec_group_iterator(group_ptr g,size_t index):
            _flat_group(g),
            _index(index)
        {}

        void increment() 
        {
            _index++;
        }

        boost::python::object __iter__() const
        {
            return boost::python::object(iterator_type(_flat_group,_index));
        }

        boost::python::object next()
        {
            if(_index == _flat_group->size())
            {
                throw(rec_group_iterator_stop());
                return boost::python::object();
            }
            auto o = (*_flat_group)[_index];
            increment();
            return boost::python::object(o);
        }

};
