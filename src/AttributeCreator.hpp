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
 * Class-template for attribute creators
 *
 * Created on: March 14, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */
#ifndef __ATTRIBUTECREATOR_HPP__
#define __ATTRIBUTECREATOR_HPP__

#include<pni/utils/Types.hpp>
#include<pni/utils/Exceptions.hpp>

using namespace pni::utils;

/*! 
\ingroup utils
\brief attribute creator class

This class creates attribute objects according to the configuration of the class.
The created attributes are instances of the template parameter AttrT. 
*/
template<typename AttrT> class AttributeCreator{
    private:
        String __n; //!< name of the field
        std::vector<size_t> __s;  //!< shape of the field
    public:
        //---------------------------------------------------------------------
        /*! \brief constructor

        \param n name of the attribute
        */
        AttributeCreator(const String &n):
            __n(n),__s(){}

        //---------------------------------------------------------------------
        /*! \brief constructor

        \param n name of the attribute
        \param s shape of the attribute
        */
        template<typename CTYPE>
            AttributeCreator(const String &n,const CTYPE &s):
            __n(n),__s(s){}

        //---------------------------------------------------------------------
        /*! \brief attribute creation
        
        Create an attribute by o. The data-type used for the attribute is 
        determined by the template parameter T. OType is the type of the object
        responsible for attribute creation.
        \throws NXAttributeError in case of errors
        \param o attribute creating object.
        \return instance of AttrT
        */
        template<typename T,typename OType> AttrT create(const OType &o) const;

        //---------------------------------------------------------------------
        /*! \brief create attribute from type string
        
        Creates an attribute below o of a data-type determined by type_str.
        \throws TypeError if type_str cannot be interpreted (unknown type-code)
        \throws NXAttributeError in case of errors during attribute creation
        \param o parent object (creating object)
        \param type_str string determining the data-type to use
        \return instance of AttrT 
        */
        template<typename OType> 
            AttrT create(const OType &o,const String &type_str) const;
};

//-----------------------------------------------------------------------------
template<typename AttrT>
template<typename T,typename OType> 
    AttrT AttributeCreator<AttrT>::create(const OType &o) const
{
    if(__s.size() == 0){
        //create a scalar attribute
        return AttrT(o.template attr<T>(__n,true));
    }else{
        //create a field with custom chunk 
        return AttrT(o.template attr<T>(__n,__s,true));
    }
}

//------------------------------------------------------------------------------
template<typename AttrT> 
template<typename OType> AttrT 
AttributeCreator<AttrT>::create(const OType &o,const String &type_code) const
{
    if(type_code == "uint8") return this->create<UInt8>(o);
    if(type_code == "int8")  return this->create<Int8>(o);
    if(type_code == "uint16") return this->create<UInt16>(o);
    if(type_code == "int16")  return this->create<Int16>(o);
    if(type_code == "uint32") return this->create<UInt32>(o);
    if(type_code == "int32")  return this->create<Int32>(o);
    if(type_code == "uint64") return this->create<UInt64>(o);
    if(type_code == "int64")  return this->create<Int64>(o);

    if(type_code == "float32") return this->create<Float32>(o);
    if(type_code == "float64") return this->create<Float64>(o);
    if(type_code == "float128") return this->create<Float128>(o);
    
    if(type_code == "complex64") return this->create<Complex32>(o);
    if(type_code == "complex128") return this->create<Complex64>(o);
    if(type_code == "complex256") return this->create<Complex128>(o);

    if(type_code == "string") return this->create<String>(o);

    //raise an exception here
    throw TypeError(EXCEPTION_RECORD,
        "Cannot create field with type-code ("+type_code+")!");
}

#endif
