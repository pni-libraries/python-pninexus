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
// Created on: Jan 31, 2018
//     Authors:
//              Eugen Wintersberger <eugen.wintersberger@desy.de>
//              Jan Kotanski <jan.kotanski@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>

namespace {

hdf5::Dimensions get_dimensions(const hdf5::dataspace::Hyperslab &self)
{
  return self.dimensions();
}

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


// hdf5::Dimensions get_points_dimensions(const hdf5::dataspace::Points &self)
// {
//   return self.dimensions();
// }

std::vector<std::vector<hsize_t>> get_coord_set(boost::python::list coord_set){

  std::vector<std::vector<hsize_t>> coords;

  for (boost::python::ssize_t i = 0, iend = len(coord_set); i < iend; ++i){
    std::vector<hsize_t> crd;
    boost::python::list coord(coord_set[i]);
    for (boost::python::ssize_t j = 0, jend = len(coord); j < jend; ++j){
      boost::python::object o = coord[j];
      boost::python::extract<hsize_t> s(o);
      if (s.check()){
        crd.push_back(s());
      }
    }
    coords.push_back(crd);
  }
  return coords;
}

std::vector<hsize_t> get_coords(boost::python::list coords){

  std::vector<hsize_t> crd;

  for (boost::python::ssize_t j = 0, jend = len(coords); j < jend; ++j){
    boost::python::object o = coords[j];
    boost::python::extract<hsize_t> s(o);
    if (s.check()){
      crd.push_back(s());
    }
  }
  return crd;
}

class DLL_EXPORT PointsWrapper : public hdf5::dataspace::Points
{
 public:
  PointsWrapper(): hdf5::dataspace::Points()
    {}
  PointsWrapper(size_t rank): hdf5::dataspace::Points(rank)
    {}
  PointsWrapper(boost::python::list coord_set):
    hdf5::dataspace::Points(get_coord_set(coord_set))
    {}
  void add_set(boost::python::list coord_set){
    hdf5::dataspace::Points::add_set(get_coord_set(coord_set));
  }
  void add(boost::python::list coords){
    hdf5::dataspace::Points::add(get_coords(coords));
  }
};


void create_selections()
{
  using namespace boost::python;
  using namespace hdf5::dataspace;

  enum_<SelectionType>("SelectionType")
      .value("NONE",SelectionType::None)
      .value("POINTS",SelectionType::Points)
      .value("HYPERSLAB",SelectionType::Hyperslab)
      .value("ALL",SelectionType::All);

  enum_<SelectionOperation>("SelectionOperation")
      .value("SET",SelectionOperation::Set)
      .value("OR",SelectionOperation::Or)
      .value("AND",SelectionOperation::And)
      .value("XOR",SelectionOperation::XOr)
      .value("NOTB",SelectionOperation::NotB)
      .value("NOTA",SelectionOperation::NotA)
      .value("APPEND",SelectionOperation::Append)
      .value("PREPEND",SelectionOperation::Prepend)
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
      .add_property("size",&Hyperslab::size)
      .add_property("type",&Hyperslab::type)
      .def("dimensions",get_dimensions)
      .def("offset",get_offset)
      .def("offset",set_entire_offset)
      .def("offset",set_individual_offset)
      .def("stride",get_stride)
      .def("stride",set_entire_stride)
      .def("stride",set_individual_stride)
      .def("count",get_count)
      .def("count",set_entire_count)
      .def("count",set_individual_count)
      .def("block",get_block)
      .def("block",set_entire_block)
      .def("block",set_individual_block)
           ;

  class_<PointsWrapper, bases<Selection>>("Points")
    .def(init<boost::python::list>(args("coord_set")))
      .add_property("rank",&PointsWrapper::rank)
      .add_property("size",&PointsWrapper::size)
      .add_property("type",&PointsWrapper::type)
      .add_property("points",&PointsWrapper::points)
      .def("dimensions", &PointsWrapper::dimensions)
    ;

  class_<View>("View")
      .def(init<Dataspace,Hyperslab>((arg("space"),arg("selection"))))
      .def(init<Dataspace>((arg("space"))))
      .add_property("size",&View::size);

}
