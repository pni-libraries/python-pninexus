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
 * Class-template for field creators.
 *
 * Created on: March 13, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __FIELDCREATOR_HPP__
#define __FIELDCREATOR_HPP__

#include<boost/python.hpp>

#include<pni/utils/Types.hpp>
#include<pni/utils/Exceptions.hpp>
#include<pni/utils/Array.hpp>

#include<pni/nx/NX.hpp>

#include "NXFieldWrapper.hpp"

using namespace pni::utils;
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
        String __n;      //!< name of the field
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
        FieldCreator(const String &n,const shape_t &s,const shape_t &cs,const object
                &filter):
            __n(n),__s(s),__cs(cs),__filter(filter){}
       
        //---------------------------------------------------------------------
        /*! \brief create field object

        This template emthod finally creates the field object. The datatype to 
        use is determined by the template parameter T. OType is the type of the
        parent object of the field.
        \throws NXFieldError in case of field related problems
        \throws NXFilterError in case of filter related errors
        \tparam T data type for which to create a field
        \param parent parent group
        \return instance of a python object
        */
        template<typename T> object create(const group_t &parent) const;

        //---------------------------------------------------------------------
        /*! \brief create field using a type string

        This is the method usually used by a client of this class to create an
        instance of an NXField object. The datatype is determined by a string.
        \throws TypeError if the datatype given by the user could no be handled
        \throws NXFieldError if field creation fails
        \throws NXFilterError if the filter object is invalid
        \param parent parent object 
        \param type_str string representing the data-type to use
        */
        object create(const group_t &parent,const String &type_str) const;
};

//-----------------------------------------------------------------------------
template<typename GTYPE>
template<typename T> 
    object FieldCreator<GTYPE>::create(const group_t &parent) const
{
    extract<NXDeflateFilter> deflate_obj(__filter);

    //check if the filter is a valid deflate filter object
    if(deflate_obj.check())
    {
        NXDeflateFilter deflate = deflate_obj();
        if(__cs.size()==0)
            return object(new field_wrapper_t(
                        parent.template create_field<T>(__n,__s,deflate)));
        else
            return object(new field_wrapper_t(
                        parent.template create_field<T>(__n,__s,__cs,deflate)));
    }
    //if the filter object is a NONE a field without filter is created
    else if(__filter.ptr() == Py_None)
        return object(new field_wrapper_t(
                    parent.template create_field<T>(__n,__s,__cs)));
    else
        throw pni::nx::NXFilterError(EXCEPTION_RECORD,
                "Invalid filter object!");
}

//------------------------------------------------------------------------------
template<typename GTYPE> object 
FieldCreator<GTYPE>::create(const group_t &parent,const String &type_code) const
{
    if(type_code == "uint8") return this->create<UInt8>(parent);
    if(type_code == "int8")  return this->create<Int8>(parent);
    if(type_code == "uint16") return this->create<UInt16>(parent);
    if(type_code == "int16")  return this->create<Int16>(parent);
    if(type_code == "uint32") return this->create<UInt32>(parent);
    if(type_code == "int32")  return this->create<Int32>(parent);
    if(type_code == "uint64") return this->create<UInt64>(parent);
    if(type_code == "int64")  return this->create<Int64>(parent);

    if(type_code == "float32") return this->create<Float32>(parent);
    if(type_code == "float64") return this->create<Float64>(parent);
    if(type_code == "float128") return this->create<Float128>(parent);
    
    if(type_code == "complex64") return this->create<Complex32>(parent);
    if(type_code == "complex128") return this->create<Complex64>(parent);
    if(type_code == "complex256") return this->create<Complex128>(parent);

    if(type_code == "string") return this->create<String>(parent);
    if(type_code == "bool")   return this->create<Bool>(parent);

    //raise an exception here
    throw TypeError(EXCEPTION_RECORD, 
            "Cannot create field with type-code ("+type_code+")!");
}

#endif
