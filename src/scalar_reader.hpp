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

//! 
//! \ingroup ioclasses  
//! \brief reads a single scalar
//! 
//! ScalarReader reads a single scalar form a readable object. For now only 
//! fields, selections, and attributes expose the appropriate interface. 
//! The class provides a static template method which reads the data and 
//! returns the result as a Python object.
//!
class scalar_reader
{
    public:
        //! 
        //! \brief read scalar data
        //!
        //! Reads scalar data of type T from a readable object and returns a 
        //! native Python object as result.
        //!
        //! \tparam T data type to read
        //! \tparam OTYPE object type where to read data from
        //! \param readable instance of OTYPE from which to read data 
        //! \return native Python object
        //!
        template<
                 typename T,
                 typename OTYPE
                > 
        static object read(const OTYPE &readable)
        {
            T value; //create a new instance where to store the data
            readable.read(value); //read data
            object o(value); //create python object
            return o;
        }
};
