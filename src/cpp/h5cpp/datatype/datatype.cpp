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
//             Eugen Wintersberger <eugen.wintersberger@desy.de>
//             Jan Kotanski <jan.kotanski@desy.de>
//
#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <cstdint>
#include <h5cpp/datatype/datatype.hpp>
#include <h5cpp/datatype/enum.hpp>
#include <h5cpp/datatype/ebool.hpp>


const boost::python::list integer_pad(const hdf5::datatype::Integer &self)
{
  std::vector<hdf5::datatype::Pad> pad_ = self.pad();
  boost::python::list padlist;
  for (auto pd: pad_){
      padlist.append(pd);
  }
  return padlist;
}

BOOST_PYTHON_MODULE(_datatype)
{
  using namespace boost::python;
  using namespace hdf5::datatype;

  //
  // setting up the documentation options
  //
  docstring_options doc_opts;
  doc_opts.disable_signatures();
  doc_opts.enable_user_defined();


  enum_<Class>("Class")
      .value("NONE",Class::NONE)
      .value("INTEGER",Class::INTEGER)
      .value("FLOAT",Class::FLOAT)
      .value("TIME",Class::TIME)
      .value("STRING",Class::STRING)
      .value("BITFIELD",Class::BITFIELD)
      .value("OPAQUE",Class::OPAQUE)
      .value("COMPOUND",Class::COMPOUND)
      .value("REFERENCE",Class::REFERENCE)
      .value("ENUM",Class::ENUM)
      .value("VARLENGTH",Class::VARLENGTH)
      .value("ARRAY",Class::ARRAY);

  enum_<Order>("Order")
      .value("LE",Order::LE)
      .value("BE",Order::BE)
      .value("VAX",Order::VAX)
      .value("NONE",Order::NONE);

  enum_<Sign>("Sign")
      .value("TWOS_COMPLEMENT",Sign::TWOS_COMPLEMENT)
      .value("UNSIGNED",Sign::UNSIGNED);

  enum_<Norm>("Norm")
      .value("IMPLIED",Norm::IMPLIED)
      .value("MSBSET",Norm::MSBSET)
      .value("NONE",Norm::NONE);

  enum_<Pad>("Pad")
      .value("ZERO",Pad::ZERO)
      .value("ONE",Pad::ONE)
      .value("BACKGROUND",Pad::BACKGROUND);

  enum_<StringPad>("StringPad")
      .value("NULLTERM",StringPad::NULLTERM)
      .value("NULLPAD",StringPad::NULLPAD)
      .value("SPACEPAD",StringPad::SPACEPAD);

  enum_<Direction>("Direction")
      .value("ASCEND",Direction::ASCEND)
      .value("DESCEND",Direction::DESCEND);

  enum_<CharacterEncoding>("CharacterEncoding")
      .value("ASCII",CharacterEncoding::ASCII)
      .value("UTF8",CharacterEncoding::UTF8);

  size_t (Datatype::*get_size)() const = &Datatype::size;
  void (Datatype::*set_size)(size_t) const = &Datatype::size;
  class_<Datatype>("Datatype")
      .add_property("type",&Datatype::get_class)
      .add_property("super",&Datatype::super)
      .def("native_type",&Datatype::native_type,(arg("dir")=Direction::ASCEND))
      .def("has_class",&Datatype::has_class)
      .add_property("size",get_size,set_size)
      .add_property("is_valid",&Datatype::is_valid)
      .def(self == Datatype())
      ;


  size_t (Integer::*get_precision)() const = &Integer::precision;
  void (Integer::*set_precision)(size_t) const = &Integer::precision;
  size_t (Integer::*get_offset)() const = &Integer::offset;
  void (Integer::*set_offset)(size_t) const = &Integer::offset;
  Order (Integer::*get_order)() const = &Integer::order;
  void (Integer::*set_order)(Order) const = &Integer::order;
  //  const std::vector<Pad> (Integer::*get_pad)() const = &Integer::pad;
  void (Integer::*set_pad)(Pad,Pad) const = &Integer::pad;
  class_<Integer,bases<Datatype>>("Integer")
      .def(init<const Datatype&>())
      .def("make_signed", &Integer::make_signed)
      .def("is_signed", &Integer::is_signed)
      .add_property("precision",get_precision,set_precision)
      .add_property("offset",get_offset,set_offset)
      .add_property("order",get_order,set_order)
      .def("pad", integer_pad)
      .def("make_pad", set_pad)
      ;

  class_<Float,bases<Datatype>>("Float")
      .def(init<const Datatype&>())
      ;


  void (String::*string_set_size)(size_t) const = &String::size;
  size_t (String::*string_get_size)() const = &String::size;
  void (String::*string_set_padding)(StringPad) = &String::padding;
  StringPad (String::*string_get_padding)() const = &String::padding;
  void (String::*string_set_encoding)(CharacterEncoding) = &String::encoding;
  CharacterEncoding (String::*string_get_encoding)() const = &String::encoding;
  class_<String,bases<Datatype>>("String")
      .def(init<const Datatype&>())
      .add_property("is_variable_length",&String::is_variable_length)
      .add_property("encoding",string_get_encoding,string_set_encoding)
      .add_property("padding",string_get_padding,string_set_padding)
      .add_property("size",string_get_size,string_set_size)
      .def("variable",&String::variable)
      .staticmethod("variable")
      .def("fixed",&String::fixed)
      .staticmethod("fixed");

  class_<Enum, bases<Datatype>>("Enum")
      .def(init<const Datatype&>())
    ;


  scope current;

  current.attr("kUInt8") = hdf5::datatype::create<uint8_t>();
  current.attr("kInt8") = hdf5::datatype::create<int8_t>();
  current.attr("kUInt16") = hdf5::datatype::create<uint16_t>();
  current.attr("kInt16")  = hdf5::datatype::create<int16_t>();
  current.attr("kUInt32") = hdf5::datatype::create<uint32_t>();
  current.attr("kInt32")  = hdf5::datatype::create<int32_t>();
  current.attr("kUInt64") = hdf5::datatype::create<uint64_t>();
  current.attr("kInt64")  = hdf5::datatype::create<int64_t>();
  current.attr("kFloat64") = hdf5::datatype::create<double>();
  current.attr("kFloat32") = hdf5::datatype::create<float>();
  current.attr("kFloat128") = hdf5::datatype::create<long double>();
  current.attr("kVariableString") = hdf5::datatype::create<std::string>();
  current.attr("kEBool") = hdf5::datatype::create<hdf5::datatype::EBool>();

  //need some functions
  def("is_bool",&is_bool);
}
