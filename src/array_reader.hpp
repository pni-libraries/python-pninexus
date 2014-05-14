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
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: May 7, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include "numpy_utils.hpp"

//! 
//! \ingroup ioclasses  
//! \brief reads a single array 
//! 
//! Reads a single array from a readable object which might be either a 
//! selection, a field, or an attribute object. The result is returned as a 
//! numpy array.
//!
class array_reader
{
    public:
        //! 
        //! \brief read array 
        //!
        //! Read a single array from the field.
        //! \tparam T data type to read
        //! \tparam OTYPE object type from which to read data
        //! \param readable instance of OTYPE from which to read data
        //! \return numpy array as Python object.
        //! 
        template<
                 typename T,
                 typename OTYPE
                > 
        static object read(const OTYPE &readable)
        {
            //create the numpy array which will store the data
            auto shape = readable.template shape<shape_t>();
            object narray = numpy::create_array(readable.type_id(),shape);

            //read data to the numpy buffer
            //we can safely use the pointer as the target array is created 
            //from the properties of the field and thus the size must match
            readable.read(numpy::get_data<T>(narray));
            return narray;
        }
};
