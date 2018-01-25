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
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

void wrap_nodes()
{
  using namespace boost::python;
  using namespace hdf5::node;

  enum_<Type>("NodeType")
      .value("UNKOWN",Type::UNKNOWN)
      .value("GROUP",Type::GROUP)
      .value("DATASET",Type::DATASET)
      .value("DATATYPE",Type::DATATYPE)
      ;

  class_<Node>("Node")
      .add_property("type",&Node::type)
      .add_property("is_valid",&Node::is_valid)
      .add_property("link",make_function(&Node::link,return_internal_reference<>()))
      .def_readonly("attributes",&Node::attributes)
      ;

  class_<Group,bases<Node>>("Group")
      ;

  class_<Dataset,bases<Node>>("Dataset")
      ;


}
