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

#include <boost/python.hpp>
#include <pni/core/types.hpp>
#include <pni/core/error.hpp>
#include <core/utils.hpp>
#include <pni/io/nexus.hpp>
#include "iterator_wrapper.hpp"

#include "errors.hpp"

class AttributeManagerWrapper
{
  private:
    hdf5::attribute::AttributeManager manager_;
  public:

    //--------------------------------------------------------------------
    AttributeManagerWrapper(const hdf5::attribute::AttributeManager &manager);

    //--------------------------------------------------------------------
    AttributeManagerWrapper(const AttributeManagerWrapper &w) = default;

    //--------------------------------------------------------------------
    bool exists(const pni::core::string &name) const;

    //--------------------------------------------------------------------
    void remove(const pni::core::string &name) const;

    //--------------------------------------------------------------------
    hdf5::attribute::Attribute create(const pni::core::string &name,
                                      const pni::core::string &type,
                                      const boost::python::object &shape,
                                      bool overwrite);

    //--------------------------------------------------------------------
    size_t size() const;

    //--------------------------------------------------------------------
    hdf5::attribute::Attribute get_by_name(const pni::core::string &name);

    //--------------------------------------------------------------------
    hdf5::attribute::Attribute get_by_index(size_t i);
    //--------------------------------------------------------------------

    boost::python::object __iter__();

};

void wrap_nxattribute_manager(const char *class_name);

