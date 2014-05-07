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
///

#pragma once

#include<boost/python.hpp>

#include<pni/core/types.hpp>
#include<pni/core/error.hpp>
#include<pni/core/arrays.hpp>

#include<pni/io/nx/nx.hpp>

#include "NXFieldWrapper.hpp"
#include "NXObjectMap.hpp"

using namespace pni::core;
using namespace boost::python;

/*! 
\ingroup utils  
\brief field creator class-template

This template generats classes whose instance are responsible for creating
NXField instances. The instance is of type FieldT. 
\tparam GTYPE group type responsible for field creation
*/
template<typename GTYPE> class FieldCreator
{
    private:
        string __n;      //!< name of the field
        shape_t __s;     //!< shape of the field
        shape_t __cs;    //!< chunk shape of the field
        object __filter; //!< name of the filter to use

        //====================private methods==================================

    public:
        //===================public types======================================
        //! group type
        typedef GTYPE group_t;
        //! field type
        typedef typename NXObjectMap<group_t>::FieldType field_t;
        //! field wrapper type
        typedef NXFieldWrapper<field_t> field_wrapper_t;
        //====================constructor======================================
        /*! \brief constructor
       
        The standard constructor for this class.
        \param n name of the field
        \param s shape of the field
        \param cs chunk shape
        \param filter filter object to use for compression
        */
        FieldCreator(const string &n,const shape_t &s,const shape_t &cs,const object
                &filter):
            __n(n),__s(s),__cs(cs),__filter(filter){}
       
        //---------------------------------------------------------------------
        /*! \brief create field object

        This template emthod finally creates the field object. The datatype to 
        use is determined by the template parameter T. OType is the type of the
        parent object of the field.
        \throws nxfield_error in case of field related problems
        \throws nxfilter_error in case of filter related errors
        \tparam T data type for which to create a field
        \param parent parent group
        \return instance of a python object
        */
        template<typename T> field_wrapper_t create(const group_t &parent) const;

        //---------------------------------------------------------------------
        /*! \brief create field using a type string

        This is the method usually used by a client of this class to create an
        instance of an NXField object. The datatype is determined by a string.
        \throws type_error if the datatype given by the user could no be handled
        \throws nxfield_error if field creation fails
        \throws nxfilter_error if the filter object is invalid
        \param parent parent object 
        \param type_str string representing the data-type to use
        */
        field_wrapper_t create(const group_t &parent,const string &type_str) const;
};

//-----------------------------------------------------------------------------
template<typename GTYPE>
template<typename T> typename FieldCreator<GTYPE>::field_wrapper_t
FieldCreator<GTYPE>::create(const group_t &parent) const
{
    extract<nxdeflate_filter> deflate_obj(__filter);

    //check if the filter is a valid deflate filter object
    field_wrapper_t wrapper;
    if(deflate_obj.check())
    {
        nxdeflate_filter deflate = deflate_obj();
        if(__cs.size()==0)
            wrapper = field_wrapper_t(
                      parent.template create_field<T>(__n,__s,deflate)
                      );
        else
            wrapper = field_wrapper_t(
                      parent.template create_field<T>(__n,__s,__cs,deflate)
                      ); 
    }
    //if the filter object is a NONE a field without filter is created
    else if(__filter.ptr() == Py_None)
        wrapper = field_wrapper_t(parent.template create_field<T>(__n,__s,__cs));
    else
        throw pni::io::nx::nxfilter_error(EXCEPTION_RECORD,
                "Invalid filter object!");

    return wrapper;
}

//------------------------------------------------------------------------------
template<typename GTYPE> typename FieldCreator<GTYPE>::field_wrapper_t
FieldCreator<GTYPE>::create(const group_t &parent,const string &type_code) const
{
    if(type_code == "uint8") return this->create<uint8>(parent);
    if(type_code == "int8")  return this->create<int8>(parent);
    if(type_code == "uint16") return this->create<uint16>(parent);
    if(type_code == "int16")  return this->create<int16>(parent);
    if(type_code == "uint32") return this->create<uint32>(parent);
    if(type_code == "int32")  return this->create<int32>(parent);
    if(type_code == "uint64") return this->create<uint64>(parent);
    if(type_code == "int64")  return this->create<int64>(parent);

    if(type_code == "float32") return this->create<float32>(parent);
    if(type_code == "float64") return this->create<float64>(parent);
    if(type_code == "float128") return this->create<float128>(parent);
    
    if(type_code == "complex64") return this->create<complex32>(parent);
    if(type_code == "complex128") return this->create<complex64>(parent);
    if(type_code == "complex256") return this->create<complex128>(parent);

    if(type_code == "string") return this->create<string>(parent);
    if(type_code == "bool")   return this->create<bool_t>(parent);

    //raise an exception here
    throw type_error(EXCEPTION_RECORD, 
            "Cannot create field with type-code ("+type_code+")!");
}

