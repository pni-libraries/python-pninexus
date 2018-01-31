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
#include "../common/io.hpp"

namespace {

void write_all(const hdf5::node::Dataset &self,const boost::python::object &data)
{
  io::write(self,data);
}

void write_with_selection(const hdf5::node::Dataset &self,
                          const boost::python::object &data,
                          const hdf5::dataspace::Selection &selection)
{

}

boost::python::object read_all(const hdf5::node::Dataset &self)
{
  return io::read(self);
}

void read_with_selection(const hdf5::node::Dataset &self,
                         boost::python::object &data,
                         const hdf5::dataspace::Selection &selection)
{

}

} //anonymous namespace

void create_dataset_wrapper()
{
  using namespace boost::python;
  using namespace hdf5::node;

  class_<Dataset,bases<Node>>("Dataset")
      .def(init<const Group&,const hdf5::Path &,
                const hdf5::datatype::Datatype &,
                const hdf5::dataspace::Dataspace &,
                const hdf5::property::LinkCreationList &,
                const hdf5::property::DatasetCreationList &,
                const hdf5::property::DatasetAccessList&>(
                (arg("parent"),arg("path"),arg("type"),arg("space"),
                    arg("lcpl")=hdf5::property::LinkCreationList(),
                    arg("dcpl")=hdf5::property::DatasetCreationList(),
                    arg("dapl")=hdf5::property::DatasetAccessList()
                    )
                ))
      .def("close",&Dataset::close)
      .def("write",write_all)
      .def("write",write_with_selection)
      .def("read",read_all)
      .def("read",read_with_selection)
      ;
}
