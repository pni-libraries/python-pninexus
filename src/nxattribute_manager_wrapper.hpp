//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Oct 30, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#pragma once

#include <pni/io/nx/nxobject_traits.hpp>
#include "errors.hpp"

using namespace pni::io::nx;

template<typename AMT>
class nxattribute_manager_wrapper
{
    public:
        typedef AMT manager_type;
        typedef typename manager_type::attribute_type attribute_type;

        typedef typename manager_type::iterator iterator;

        typedef nxattribute_manager_wrapper<manager_type> wrapper_type;

    private:
        manager_type _manager;
        
        size_t _index;
        attribute_type _attr;
    public:
       
        //--------------------------------------------------------------------
        explicit nxattribute_manager_wrapper(const manager_type &m):
            _manager(m),
            _index(0),
            _attr()
        {
            if(_manager.size()) _attr = _manager[_index];
        }

        //--------------------------------------------------------------------
        explicit nxattribute_manager_wrapper(const wrapper_type &w):
            _manager(w._manager),
            _index(0),
            _attr()
        {
            if(_manager.size()) _attr= _manager[_index];
        }

        //--------------------------------------------------------------------
        size_t size() const 
        {
            return _manager.size();
        }

        //--------------------------------------------------------------------
        attribute_type get_by_name(const string &name)
        {
            return _manager[name];
        }

        //--------------------------------------------------------------------
        attribute_type get_by_index(size_t i)
        {
            return _manager[i];
        }
        //--------------------------------------------------------------------
        
        object __iter__()
        {
            return object(this);
        }


        void increment()
        {
            _index++;
            if(_index < _manager.size()) _attr = _manager[_index];
        }

        attribute_type next()
        {
            //check if iteration is still possible
            if(_index >= _manager.size())
            {
                //raise exception here
                throw(AttributeIteratorStop());
                return(attribute_type());
            }

            attribute_type attr(_attr);
            this->increment();

            return attr;
        }

};

template<typename AMT> void wrap_nxattribute_manager(const string &name)
{
    typedef nxattribute_manager_wrapper<AMT> wrapper_type;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    class_<wrapper_type>(name.c_str(),init<const AMT&>())
        .add_property("size",&wrapper_type::size)   
        .def("__getitem__",&wrapper_type::get_by_name)
        .def("__getitem__",&wrapper_type::get_by_index)
        .def("__len__",&wrapper_type::size)
        .def("__iter__",&wrapper_type::__iter__)
        .def("next",&wrapper_type::next)
        .def("increment",&wrapper_type::increment)
        ;
#pragma GCC diagnostic pop
}

