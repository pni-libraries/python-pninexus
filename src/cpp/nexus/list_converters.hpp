//
// (c) Copyright 2018 DESY
//
// This file is part of python-pninexus.
//
// python-pninexus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pninexus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
//  Created on: Feb 8, 2018
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <pni/nexus.hpp>

//!
//! @brief convert a nexus GroupList to a tuple
//!
struct GroupListToTuple
{
    GroupListToTuple();
    static PyObject *convert(const pni::nexus::GroupList &group_list);
};

//!
//! @brief convert a nexus DatasetList to a tuple
//!
struct DatasetListToTuple
{
    DatasetListToTuple();
    static PyObject *convert(const pni::nexus::DatasetList &dataset_list);
};

//!
//! @brief convert a nexus NodeList to a tuple
//!
struct NodeListToTuple
{
    NodeListToTuple();
    static PyObject *convert(const pni::nexus::NodeList &node_list);
};

//!
//! @brief convert a nexus AttributeList to a tuple
//!
struct AttributeListToTuple
{
    AttributeListToTuple();
    static PyObject *convert(const pni::nexus::AttributeList &attribute_list);
};

//!
//! @brief convert a nexus PathObjectList to a tuple
//!
struct PathObjectListToTuple
{
    PathObjectListToTuple();
    static PyObject *convert(const pni::nexus::PathObjectList &pathobject_list);
};
