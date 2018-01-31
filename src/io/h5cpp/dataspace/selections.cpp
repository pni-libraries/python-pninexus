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
// Created on: Jan 31, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

namespace {

hdf5::Dimensions get_offset(const hdf5::dataspace::Hyperslab &self)
{
  return self.offset();
}

hdf5::Dimensions get_stride(const hdf5::dataspace::Hyperslab &self)
{
  return self.stride();
}

hdf5::Dimensions get_count(const hdf5::dataspace::Hyperslab &self)
{
  return self.count();
}

hdf5::Dimensions get_block(const hdf5::dataspace::Hyperslab &self)
{
  return self.block();
}

//struct SelectionWrapper : Selection,wrapper<Selection>
//{
//  void apply(const hdf5::dataspace::Dataspace &space,hdf5::dataspace::SelectionOperation ops)
//  {
//    this->get_override("apply")(space,ops);
//  }
//};

} // anonymous namespace

void create_selections()
{
  using namespace boost::python;
  using namespace hdf5::dataspace;

  enum_<SelectionType>("SelectionType")
      .value("NONE",SelectionType::NONE)
      .value("POINTS",SelectionType::POINTS)
      .value("HYPERSLAB",SelectionType::HYPERSLAB)
      .value("ALL",SelectionType::ALL);

  enum_<SelectionOperation>("SelectionOperation")
      .value("SET",SelectionOperation::SET)
      .value("OR",SelectionOperation::OR)
      .value("AND",SelectionOperation::AND)
      .value("XOR",SelectionOperation::XOR)
      .value("NOTB",SelectionOperation::NOTB)
      .value("NOTA",SelectionOperation::NOTA)
      .value("APPEND",SelectionOperation::APPEND)
      .value("PREPEND",SelectionOperation::PREPEND)
      ;


  class_<Selection,boost::noncopyable>("Selection",no_init);

  void (Hyperslab::*set_individual_offset)(size_t,size_t) = &Hyperslab::offset;
  void (Hyperslab::*set_entire_offset)(const hdf5::Dimensions &) = &Hyperslab::offset;
  void (Hyperslab::*set_individual_stride)(size_t,size_t) = &Hyperslab::stride;
  void (Hyperslab::*set_entire_stride)(const hdf5::Dimensions &) = &Hyperslab::stride;
  void (Hyperslab::*set_individual_count)(size_t,size_t) = &Hyperslab::count;
  void (Hyperslab::*set_entire_count)(const hdf5::Dimensions &) = &Hyperslab::count;
  void (Hyperslab::*set_individual_block)(size_t,size_t) = &Hyperslab::block;
  void (Hyperslab::*set_entire_block)(const hdf5::Dimensions &) = &Hyperslab::block;
  class_<Hyperslab,bases<Selection>>("Hyperslab")
      .def(init<hdf5::Dimensions,hdf5::Dimensions,hdf5::Dimensions,hdf5::Dimensions>
           ((arg("offset"),arg("block"),arg("count"),arg("stride")))
           )
      .def(init<hdf5::Dimensions,hdf5::Dimensions>
           ((arg("offset"),arg("block")))
           )
      .def(init<hdf5::Dimensions,hdf5::Dimensions,hdf5::Dimensions>
           ((arg("offset"),arg("count"),arg("stride")))
           )
      .add_property("rank",&Hyperslab::rank)
      .add_property("offset",&get_offset,set_entire_offset)
      .def("offset",set_individual_offset)
      .add_property("stride",&get_stride,set_entire_stride)
      .def("stride",set_individual_stride)
      .add_property("count",&get_count,set_entire_count)
      .def("count",set_individual_count)
      .add_property("block",&get_block,set_entire_block)
      .def("block",set_individual_block)
           ;

  class_<View>("View")
      .def(init<Dataspace,Hyperslab>((arg("space"),arg("selection"))))
      .add_property("size",&View::size);

}