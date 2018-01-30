//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Oct 8, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <h5cpp/hdf5.hpp>
#include <pni/io/nexus.hpp>
#include <pni/core/error.hpp>
#include <boost/python/extract.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
#define NO_IMPORT_ARRAY
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include "numpy.hpp"

namespace hdf5  {
namespace dataspace {
    
template<> class TypeTrait<numpy::ArrayAdapter>
{
  public:
    using DataspaceType = Simple;

    static DataspaceType create(const numpy::ArrayAdapter &array)
    {
      return Simple(array.dimensions());
    }

    static void *ptr(numpy::ArrayAdapter &array)
    {
      return array.data();
    }

    static const void *cptr(const numpy::ArrayAdapter &array)
    {
      return array.data();
    }

};
    
} // namespace dataspace

namespace datatype {

template<> class TypeTrait<numpy::ArrayAdapter>
{
  public:
    using TypeClass = Datatype;

    static TypeClass create(const numpy::ArrayAdapter &array)
    {
      using pni::io::nexus::DatatypeFactory;

      return DatatypeFactory::create(array.type_id());
    }
};

} // namespace datatype


} // namespace hdf5
