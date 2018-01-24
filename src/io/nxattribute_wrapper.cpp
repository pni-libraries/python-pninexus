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
// Created on: Sep 17, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "nxattribute_wrapper.hpp"
#include "h5_numpy_support.hpp"
#include "io_operations.hpp"

AttributeWrapper::AttributeWrapper(const hdf5::attribute::Attribute &a):
  attribute_(a)
{}

AttributeWrapper::AttributeWrapper(hdf5::attribute::Attribute &&a):
    attribute_(std::move(a))
{}

boost::python::tuple AttributeWrapper::shape() const
{
    hdf5::Dimensions current_dims = pni::io::nexus::get_dimensions(attribute_);
    return boost::python::tuple(Container2List(current_dims));
}

pni::core::string AttributeWrapper::type_id() const
{
    pni::core::type_id_t tid = pni::io::nexus::get_type_id(attribute_);
    return numpy::type_str(tid);
}

void AttributeWrapper::close()
{
  H5Aclose(static_cast<hid_t>(attribute_));
}

bool AttributeWrapper::is_valid() const
{
  return attribute_.is_valid();
}

std::string AttributeWrapper::name() const
{
  return attribute_.name();
}

boost::python::object AttributeWrapper::read() const
{
  using namespace boost::python;

  //read all data to a numpy array
  object np_array = numpy::create_array(pni::io::nexus::get_type_id(attribute_),
                                        pni::io::nexus::get_dimensions(attribute_));
  NumpyArray np_guard{(PyArrayObject*)np_array.ptr()};
  attribute_.read(np_guard);

  //if the data item is a scalar we return only this single scalar value.
  if(numpy::get_size(np_array)==1)
    np_array = get_first_element(np_array);

  return np_array;
}

void AttributeWrapper::write(const boost::python::object &o) const
{
  if(numpy::is_array(o))
    attribute_.write(NumpyArray{(PyArrayObject*)o.ptr()});
  else
    attribute_.write(NumpyArray{(PyArrayObject*)numpy::to_numpy_array(o).ptr()});
}

boost::python::object AttributeWrapper::__getitem__(const boost::python::object &t)
{
    using namespace boost::python;

    object np_array = numpy::create_array(pni::io::nexus::get_type_id(attribute_),
                                          pni::io::nexus::get_dimensions(attribute_));
    NumpyArray array_guard{(PyArrayObject*)np_array.ptr()};
    attribute_.read(array_guard);

    return np_array[t];
}

void AttributeWrapper::__setitem__(const boost::python::object &t,
                                   const boost::python::object &o)
{
    using namespace boost::python;

    object np_array = numpy::create_array(pni::io::nexus::get_type_id(attribute_),
                                          pni::io::nexus::get_dimensions(attribute_));
    NumpyArray array_guard{(PyArrayObject*)np_array.ptr()};
    attribute_.read(array_guard);
    np_array[t] =o;
    attribute_.write(array_guard);
}


std::string AttributeWrapper::path() const
{
  using pni::io::nexus::get_path;
  using pni::io::nexus::Path;
  return Path::to_string(get_path(attribute_));
}

size_t AttributeWrapper::size() const
{
  hdf5::dataspace::Dataspace space = attribute_.dataspace();
  return space.size();
}

boost::python::object AttributeWrapper::parent() const
{
    //return *attribute_.parent_link();
  return boost::python::object();
}

std::string AttributeWrapper::filename() const
{
  return attribute_.parent_link().target().file_path().string();
}


PyObject *AttributeToPythonObject::convert(const hdf5::attribute::Attribute &attribute)
{
    using namespace boost::python;

    return incref(object(AttributeWrapper(attribute)).ptr());
}

//=============================================================================
static const char __attribute_shape_docstr[] =
"Read only property providing the shape of the attribute as tuple.\n";

static const char __attribute_dtype_docstr[] =
"Read only property providing the data-type of the attribute as numpy\n"
"type-code";

static const char __attribute_valid_docstr[] =
"Read only property returning :py:const:`True` if the attribute is a "
"valid NeXus object";

static const char __attribute_name_docstr[] =
"A read only property providing the name of the attribute as a string.";

static const char __attribute_close_docstr[] =
"Class method to close an open attribute.";

static const char __attribute_write_docstr[] =
"Write attribute data \n"
"\n"
"Writes entire attribute data to disk. The argument passed to this "
"method is either a single scalar object or an instance of a numpy "
"array.\n"
"\n"
":param numpy.ndarray data: attribute data to write\n"
;

static const pni::core::string nxattribute_read_doc =
"Read entire attribute \n"
"\n"
"Reads all data from the attribute and returns it either as a single \n"
"scalar value or as an instance of a numpy array.\n"
"\n"
":return: attribute data\n"
":rtype: instance of numpy.ndarray or a scalar native Python type\n"
;

static const pni::core::string nxattribute_path_doc =
"Read only property returning the NeXus path for this attribute\n";

static const pni::core::string nxattribute_parent_doc =
"Read only property returning the parent object of this attribute\n";

static const pni::core::string nxattribute_size_doc =
"Read only property returing the number of elements this attribute holds\n";

static const pni::core::string nxattribute_filename_doc =
"Read only property returning the name of the file the attribute belongs to\n";

void wrap_nxattribute(const char *class_name)
{
    using namespace boost::python;

    class_<AttributeWrapper>(class_name)
        .add_property("dtype",&AttributeWrapper::type_id,__attribute_dtype_docstr)
        .add_property("shape",&AttributeWrapper::shape,__attribute_shape_docstr)
        .add_property("size",&AttributeWrapper::size,nxattribute_size_doc.c_str())
        .add_property("filename",&AttributeWrapper::filename,nxattribute_filename_doc.c_str())
        .add_property("name",&AttributeWrapper::name,__attribute_name_docstr)
        .add_property("parent",&AttributeWrapper::parent,nxattribute_parent_doc.c_str())
        .add_property("is_valid",&AttributeWrapper::is_valid,__attribute_valid_docstr)
        .add_property("path",&AttributeWrapper::path,nxattribute_path_doc.c_str())
        .def("close",&AttributeWrapper::close,__attribute_close_docstr)
        .def("read",&AttributeWrapper::read,nxattribute_read_doc.c_str())
        .def("write",&AttributeWrapper::write,__attribute_write_docstr)
        .def("__getitem__",&AttributeWrapper::__getitem__)
        .def("__setitem__",&AttributeWrapper::__setitem__)
        ;

}

