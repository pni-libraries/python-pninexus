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
#include <pni/core/types.hpp>
#include <pni/core/error.hpp>
#include <core/utils.hpp>

using namespace pni::io::nx;
using namespace pni::core;

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
    public:
       
        //--------------------------------------------------------------------
        explicit nxattribute_manager_wrapper(const manager_type &m):
            _manager(m),
            _index(0)
        { }

        //--------------------------------------------------------------------
        explicit nxattribute_manager_wrapper(const wrapper_type &w):
            _manager(w._manager),
            _index(w._index)
        { }

        //--------------------------------------------------------------------
        bool exists(const string &name) const
        {
            return _manager.exists(name);
        }

        //--------------------------------------------------------------------
        void remove(const string &name) const
        {
            _manager.remove(name);
        }

        //--------------------------------------------------------------------
        attribute_type create(const string &name,const string &type,
                              const object &shape,
                              bool overwrite)
        {
            auto s = Tuple2Container<shape_t>(tuple(shape));

            if(s.empty())
                s = shape_t{1};
            
            type_id_t tid = type_id_from_str(type);

            if(tid == type_id_t::UINT8)
                return _manager.template create<uint8>(name,s,overwrite);
            else if(tid == type_id_t::INT8)
                return _manager.template create<int8>(name,s,overwrite);
            else if(tid == type_id_t::UINT16)
                return _manager.template create<uint16>(name,s,overwrite);
            else if(tid == type_id_t::INT16)
                return _manager.template create<int16>(name,s,overwrite);
            else if(tid == type_id_t::UINT32)
                return _manager.template create<uint32>(name,s,overwrite);
            else if(tid == type_id_t::INT32)
                return _manager.template create<int32>(name,s,overwrite);
            else if(tid == type_id_t::UINT64)
                return _manager.template create<uint64>(name,s,overwrite);
            else if(tid == type_id_t::INT64)
                return _manager.template create<int64>(name,s,overwrite);
            else if(tid == type_id_t::FLOAT32)
                return _manager.template create<float32>(name,s,overwrite);
            else if(tid == type_id_t::FLOAT64)
                return _manager.template create<float64>(name,s,overwrite);
            else if(tid == type_id_t::FLOAT128)
                return _manager.template create<float128>(name,s,overwrite);
            else if(tid == type_id_t::COMPLEX32)
                return _manager.template create<complex32>(name,s,overwrite);
            else if(tid == type_id_t::COMPLEX64)
                return _manager.template create<complex64>(name,s,overwrite);
            else if(tid == type_id_t::COMPLEX128)
                return _manager.template create<complex128>(name,s,overwrite);
            else if(tid == type_id_t::BOOL)
                return _manager.template create<bool_t>(name,s,overwrite);
            else if(tid == type_id_t::STRING)
                return _manager.template create<string>(name,s,overwrite);
            else 
                type_error(EXCEPTION_RECORD,"Unkonwn type string!");

            return attribute_type(); //just to make the compiler happy
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
            //we return by value here and thus create a new object anyhow
            return object(this);
        }

        //--------------------------------------------------------------------
        void increment()
        {
            _index++;
        }

        //--------------------------------------------------------------------
        attribute_type next()
        {
            //check if iteration is still possible
            if(_index >= _manager.size())
            {
                //raise exception here
                throw(AttributeIteratorStop());
                return(attribute_type());
            }

            attribute_type attr(_manager[_index]);
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
        .def("create",&wrapper_type::create,("name","type",arg("shape")=list(),
                     arg("overwrite")=false))
        .def("remove",&wrapper_type::remove)
        .def("exists",&wrapper_type::exists)
        ;
#pragma GCC diagnostic pop
}

