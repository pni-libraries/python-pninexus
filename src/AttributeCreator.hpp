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
// along with pyton-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: March 14, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include<pni/core/types.hpp>
#include<pni/core/error.hpp>

using namespace pni::core;

/*! 
\ingroup utils
\brief attribute creator class

This class creates attribute objects according to the configuration of the class.
The created attributes are instances of the template parameter AttrT. 
*/
template<typename ATTRT> class AttributeCreator
{
    private:
        string __n; //!< name of the field
        std::vector<size_t> __s;  //!< shape of the field
    public:
        //---------------------------------------------------------------------
        /*! \brief constructor

        \param n name of the attribute
        */
        AttributeCreator(const string &n):__n(n),__s(){}

        //---------------------------------------------------------------------
        /*! \brief constructor

        \param n name of the attribute
        \param s shape of the attribute
        */
        template<typename CTYPE>
            AttributeCreator(const string &n,const CTYPE &s): __n(n),__s(s){}

        //---------------------------------------------------------------------
        /*! \brief attribute creation
        
        Create an attribute by o. The data-type used for the attribute is 
        determined by the template parameter T. OType is the type of the object
        responsible for attribute creation.
        \throws nxattribute_error in case of errors
        \param o attribute creating object.
        \return instance of AttrT
        */
        template<typename T,typename OType> ATTRT create(const OType &o) const;

        //---------------------------------------------------------------------
        /*! \brief create attribute from type string
        
        Creates an attribute below o of a data-type determined by type_str.
        \throws type_error if type_str cannot be interpreted (unknown type-code)
        \throws nxattribute_error in case of errors during attribute creation
        \param o parent object (creating object)
        \param type_str string determining the data-type to use
        \return instance of AttrT 
        */
        template<typename OTYPE> 
            ATTRT create(const OTYPE &o,const string &type_str) const;
};

//-----------------------------------------------------------------------------
template<typename ATTRT>
template<typename T,typename OTYPE> 
    ATTRT AttributeCreator<ATTRT>::create(const OTYPE &o) const
{
    if(__s.size() == 0)
        //create a scalar attribute
        return ATTRT(o.template attr<T>(__n,true));
    else
        //create a field with custom chunk 
        return ATTRT(o.template attr<T>(__n,__s,true));
    
}

//------------------------------------------------------------------------------
template<typename ATTRT> 
template<typename OTYPE> ATTRT 
AttributeCreator<ATTRT>::create(const OTYPE &o,const string &type_code) const
{
    if(type_code == "uint8") return this->create<uint8>(o);
    if(type_code == "int8")  return this->create<int8>(o);
    if(type_code == "uint16") return this->create<uint16>(o);
    if(type_code == "int16")  return this->create<int16>(o);
    if(type_code == "uint32") return this->create<uint32>(o);
    if(type_code == "int32")  return this->create<int32>(o);
    if(type_code == "uint64") return this->create<uint64>(o);
    if(type_code == "int64")  return this->create<int64>(o);

    if(type_code == "float32") return this->create<float32>(o);
    if(type_code == "float64") return this->create<float64>(o);
    if(type_code == "float128") return this->create<float128>(o);
    
    if(type_code == "complex64") return this->create<complex32>(o);
    if(type_code == "complex128") return this->create<complex64>(o);
    if(type_code == "complex256") return this->create<complex128>(o);

    if(type_code == "string") return this->create<string>(o);
    if(type_code == "bool") return this->create<bool_t>(o);

    //raise an exception here
    throw type_error(EXCEPTION_RECORD,
        "Cannot create field with type-code ("+type_code+")!");
}

