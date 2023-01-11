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
#include <h5cpp/contrib/nexus/ebool.hpp>
#include <h5cpp/contrib/stl/stl.hpp>


const boost::python::tuple integer_get_pad(const hdf5::datatype::Integer &self)
{
  std::vector<hdf5::datatype::Pad> pad_ = self.pad();
  if(pad_.size() != 2)
    throw std::runtime_error("Object is not a two element list");
  return boost::python::make_tuple(pad_[0],pad_[1]);
}

void integer_set_pad(const hdf5::datatype::Integer &self, const boost::python::object pad)
{
  hdf5::datatype::Pad lp = hdf5::datatype::Pad::Zero;
  hdf5::datatype::Pad mp = hdf5::datatype::Pad::Zero;
  if(boost::python::len(pad) != 2)
    throw std::runtime_error("Object is not a two element list");
  boost::python::object lo = pad[0];
  boost::python::extract<hdf5::datatype::Pad> ls(lo);
  if (ls.check())
    lp = ls();
  boost::python::object mo = pad[1];
  boost::python::extract<hdf5::datatype::Pad> ms(mo);
  if (ms.check())
    mp = ms();
  self.pad(lp,mp);

}

const boost::python::tuple float_get_pad(const hdf5::datatype::Float &self)
{
  std::vector<hdf5::datatype::Pad> pad_ = self.pad();
  if(pad_.size() != 2)
    throw std::runtime_error("Object is not a two element list");
  return boost::python::make_tuple(pad_[0],pad_[1]);
}

void float_set_pad(const hdf5::datatype::Float &self, const boost::python::object pad)
{
  hdf5::datatype::Pad lp = hdf5::datatype::Pad::Zero;
  hdf5::datatype::Pad mp = hdf5::datatype::Pad::Zero;
  if(boost::python::len(pad) != 2)
    throw std::runtime_error("Object is not a two element list");
  boost::python::object lo = pad[0];
  boost::python::extract<hdf5::datatype::Pad> ls(lo);
  if (ls.check())
    lp = ls();
  boost::python::object mo = pad[1];
  boost::python::extract<hdf5::datatype::Pad> ms(mo);
  if (ms.check())
    mp = ms();
  self.pad(lp,mp);
}

const boost::python::tuple float_get_fields(const hdf5::datatype::Float &self)
{
  std::vector<size_t> fields_ = self.fields();
  if(fields_.size() != 5)
    throw std::runtime_error("Object is not a five element list");
  return boost::python::make_tuple(fields_[0],fields_[1],fields_[2],
				   fields_[3],fields_[4]);
}


void float_set_fields(const hdf5::datatype::Float &self, const boost::python::object fields)
{
  if(boost::python::len(fields) != 5)
    throw std::runtime_error("Object is not a five element list");
  std::vector<size_t> fields_;
  for (boost::python::ssize_t i = 0, end = boost::python::len(fields); i < end; ++i){
    boost::python::object o = fields[i];
    boost::python::extract<size_t> s(o);
    if (s.check()){
      fields_.push_back(s());
    }
  }
  if(fields_.size() != 5)
    throw std::runtime_error("Object is not a five element list");
  self.fields(fields_[0],fields_[1],fields_[2],fields_[3],fields_[4]);
}

hdf5::datatype::Datatype compound_name_getitem(const hdf5::datatype::Compound &self, const std::string &name)
{
  return self[name];
}

hdf5::datatype::Datatype compound_index_getitem(const hdf5::datatype::Compound &self, size_t index)
{
  return self[index];
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
      .value("NONE",Class::None)
      .value("INTEGER",Class::Integer)
      .value("FLOAT",Class::Float)
      .value("TIME",Class::Time)
      .value("STRING",Class::String)
      .value("BITFIELD",Class::BitField)
      .value("OPAQUE",Class::Opaque)
      .value("COMPOUND",Class::Compound)
      .value("REFERENCE",Class::Reference)
      .value("ENUM",Class::Enum)
      .value("VARLENGTH",Class::VarLength)
      .value("ARRAY",Class::Array);

  enum_<Order>("Order")
      .value("LE",Order::LE)
      .value("BE",Order::BE)
      .value("VAX",Order::Vax)
      .value("NONE",Order::None);

  enum_<Sign>("Sign")
      .value("TWOS_COMPLEMENT",Sign::TwosComplement)
      .value("UNSIGNED",Sign::Unsigned);

  enum_<Norm>("Norm")
      .value("IMPLIED",Norm::Implied)
      .value("MSBSET",Norm::MSBSet)
      .value("NONE",Norm::None);

  enum_<Pad>("Pad")
      .value("ZERO",Pad::Zero)
      .value("ONE",Pad::One)
      .value("BACKGROUND",Pad::Background);

  enum_<StringPad>("StringPad")
      .value("NULLTERM",StringPad::NullTerm)
      .value("NULLPAD",StringPad::NullPad)
      .value("SPACEPAD",StringPad::SpacePad);

  enum_<Direction>("Direction")
      .value("ASCEND",Direction::Ascend)
      .value("DESCEND",Direction::Descend);

  enum_<CharacterEncoding>("CharacterEncoding")
      .value("ASCII",CharacterEncoding::ASCII)
      .value("UTF8",CharacterEncoding::UTF8);

  size_t (Datatype::*get_size)() const = &Datatype::size;
  void (Datatype::*set_size)(size_t) const = &Datatype::size;
  class_<Datatype>("Datatype")
      .add_property("type",&Datatype::get_class)
      .add_property("super",&Datatype::super)
      .def("native_type",&Datatype::native_type,(arg("dir")=Direction::Ascend))
      .def("has_class",&Datatype::has_class)
      .add_property("size",get_size,set_size)
      .add_property("is_valid",&Datatype::is_valid)
      .def(self == Datatype())
      ;


  size_t (Integer::*integer_get_precision)() const = &Integer::precision;
  void (Integer::*integer_set_precision)(size_t) const = &Integer::precision;
  size_t (Integer::*integer_get_offset)() const = &Integer::offset;
  void (Integer::*integer_set_offset)(size_t) const = &Integer::offset;
  Order (Integer::*integer_get_order)() const = &Integer::order;
  void (Integer::*integer_set_order)(Order) const = &Integer::order;
  class_<Integer,bases<Datatype>>("Integer")
      .def(init<const Datatype&>())
      .def("make_signed", &Integer::make_signed)
      .def("is_signed", &Integer::is_signed)
      .add_property("precision",integer_get_precision,integer_set_precision)
      .add_property("offset",integer_get_offset,integer_set_offset)
      .add_property("order",integer_get_order,integer_set_order)
      .add_property("pad", integer_get_pad, integer_set_pad)
      ;

  size_t (Float::*float_get_precision)() const = &Float::precision;
  void (Float::*float_set_precision)(size_t) const = &Float::precision;
  size_t (Float::*float_get_offset)() const = &Float::offset;
  void (Float::*float_set_offset)(size_t) const = &Float::offset;
  Order (Float::*float_get_order)() const = &Float::order;
  void (Float::*float_set_order)(Order) const = &Float::order;
  size_t (Float::*float_get_ebias)() const = &Float::ebias;
  void (Float::*float_set_ebias)(size_t) const = &Float::ebias;
  Norm (Float::*float_get_norm)() const = &Float::norm;
  void (Float::*float_set_norm)(Norm) const = &Float::norm;
  Pad (Float::*float_get_inpad)() const = &Float::inpad;
  void (Float::*float_set_inpad)(Pad) const = &Float::inpad;
  class_<Float,bases<Datatype>>("Float")
      .def(init<const Datatype&>())
      .add_property("precision",float_get_precision,float_set_precision)
      .add_property("offset",float_get_offset,float_set_offset)
      .add_property("order",float_get_order,float_set_order)
      .add_property("pad", float_get_pad, float_set_pad)
      .add_property("fields", float_get_fields, float_set_fields)
      .add_property("ebias",float_get_ebias,float_set_ebias)
      .add_property("norm",float_get_norm,float_set_norm)
      .add_property("inpad",float_get_inpad,float_set_inpad)
      ;

  size_t (Compound::*compound_name_field_offset)(const std::string&) const = &Compound::field_offset;
  size_t (Compound::*compound_index_field_offset)(size_t) const = &Compound::field_offset;
  Class (Compound::*compound_name_field_class)(const std::string&) const = &Compound::field_class;
  Class (Compound::*compound_index_field_class)(size_t) const = &Compound::field_class;
  class_<Compound,bases<Datatype>>("Compound")
    .def(init<const Datatype&>())
    .add_property("number_of_fields",&Compound::number_of_fields)
    .def("field_name", &Compound::field_name)
    .def("field_index",&Compound::field_index)
    .def("field_offset",compound_name_field_offset)
    .def("field_offset",compound_index_field_offset)
    .def("field_class",compound_name_field_class)
    .def("field_class",compound_index_field_class)
    .def("_getitem", compound_name_getitem)
    .def("_getitem", compound_index_getitem)
    .def("pack",&Compound::pack)
    .def("insert",&Compound::insert)
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
  current.attr("kFloat16") = hdf5::datatype::create<hdf5::datatype::float16_t>();
  current.attr("kFloat32") = hdf5::datatype::create<float>();
  current.attr("kFloat64") = hdf5::datatype::create<double>();
  current.attr("kFloat128") = hdf5::datatype::create<long double>();
  current.attr("kComplex32") = hdf5::datatype::create<std::complex<hdf5::datatype::float16_t>>();
  current.attr("kComplex64") = hdf5::datatype::create<std::complex<float>>();
  current.attr("kComplex128") = hdf5::datatype::create<std::complex<double>>();
  current.attr("kComplex256") = hdf5::datatype::create<std::complex<long double>>();
  current.attr("kVariableString") = hdf5::datatype::create<std::string>();
  current.attr("kEBool") = hdf5::datatype::create<hdf5::datatype::EBool>();

  //need some functions
  def("is_bool",&is_bool);
}
