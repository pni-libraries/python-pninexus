//
// (c) Copyright 2018 DESY
//
// This file is part of python-pni.
//
// python-pni is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pni is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 25, 2018
//     Authors:
//             Jan Kotanski <jan.kotanski@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <cstdint>
#include <h5cpp/datatype/datatype.hpp>
#include <h5cpp/datatype/enum.hpp>


namespace hdf5
{
  namespace datatype
  {
    //!
    //! \brief enumeration bool type
    //!
    enum EBool : int8_t
    {
      FALSE = 0, //!< indicates a false value
      TRUE = 1   //!< indicates a true value
    };

    template<>
    class TypeTrait<datatype::EBool> {
    public:
      using TypeClass = datatype::Enum;
      using Type = datatype::EBool;

      static TypeClass create(const Type & = Type()) {
	auto type = TypeClass::create(Type());
	type.insert("FALSE", Type::FALSE);
	type.insert("TRUE", Type::TRUE);
	return type;
      }
    };

    //!
    //! @brief check if Enum is EBool
    //!
    //! @param DataType object
    //! @return if Enum is EBool flag
    //!
    bool is_bool(const Datatype & dtype){
      if(dtype.get_class() == Class::ENUM){
	auto etype = datatype::Enum(dtype);
	int s = etype.number_of_values();
	if(s != 2){
	  return false;
	}
	if(etype.name(0) != "FALSE"){
	  return false;
	}
	if(etype.name(1) != "TRUE"){
	  return false;
	}
	return true;
      }
      else{
	return false;
      }
    }

  }
}


