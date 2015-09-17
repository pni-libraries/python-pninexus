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

#include <boost/python.hpp>

//! 
//! \ingroup ioclasses
//! \brief write scalar data
//! 
//! Writes a scalar value from a Python object to a writeable object. 
//!
class scalar_writer
{
    private:
        template<
                 typename T,
                 typename WTYPE
                >
        static void single_scalar(const WTYPE &writeable,
                                  const boost::python::object &o)
        {
            using namespace boost::python;

            T value = extract<T>(o);
            writeable.write(value);
        }

        //--------------------------------------------------------------------
        template<
                 typename T,
                 typename WTYPE
                >
        static void broadcast_scalar(const WTYPE &writeable,
                                     const boost::python::object &o)
        {
            using namespace boost::python;

            typedef pni::core::dynamic_array<T> array_type;

            auto shape = writeable.template shape<pni::core::shape_t>();
            auto data = array_type::create(shape);

            std::fill(data.begin(),data.end(),extract<T>(o));
            writeable.write(data);
        }
    public:
        //! 
        //! \brief write scalar data
        //!
        //! Writes scalar data from object o to writable.
        //! \throws ShapeMissmatchError if o is not a scalar object
        //! \throws TypeError if type conversion fails
        //! \tparam T data type to write
        //! \tparam WTYPE writeable type
        //! \param writeable instance of WTYPE where to store data
        //! \param o object form which to write data
        //!
        template<
                 typename T,
                 typename WTYPE
                > 
        static void write(const WTYPE &writeable, 
                          const boost::python::object &o)
        {
            if(writeable.size()==1)
                single_scalar<T>(writeable,o);
            else
                broadcast_scalar<T>(writeable,o);
        }
};

