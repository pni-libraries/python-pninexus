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

#include "numpy.hpp"


namespace {
  using pni::core::type_id_t;

  //!
  //! \ingroup devel_doc
  //! \brief map between type_id_t and numpy types
  //!
  //! A static map between `libpnicore` `type_id_t` values and `numpy`
  //! data types.
  static const std::map<type_id_t,int> type_id2numpy_id = {
      {type_id_t::UINT8,  NPY_UINT8},  {type_id_t::INT8,   NPY_INT8},
      {type_id_t::UINT16, NPY_UINT16}, {type_id_t::INT16,  NPY_INT16},
      {type_id_t::UINT32, NPY_UINT32}, {type_id_t::INT32,  NPY_INT32},
      {type_id_t::UINT64, NPY_UINT64}, {type_id_t::INT64,  NPY_INT64},
      {type_id_t::FLOAT32, NPY_FLOAT}, {type_id_t::FLOAT64, NPY_DOUBLE},
      {type_id_t::FLOAT128,NPY_LONGDOUBLE},
      {type_id_t::COMPLEX32, NPY_CFLOAT},
      {type_id_t::COMPLEX64, NPY_CDOUBLE},
      {type_id_t::COMPLEX128,NPY_CLONGDOUBLE},
#if PY_MAJOR_VERSION >= 3
      {type_id_t::STRING,NPY_UNICODE},
#else
      {type_id_t::STRING,NPY_STRING},
#endif
      {type_id_t::BOOL,  NPY_BOOL}
  };

  static const std::map<int,type_id_t> numpy_id2type_id = {
      {NPY_UINT8, type_id_t::UINT8},  {NPY_INT8, type_id_t::INT8},
      {NPY_UINT16,type_id_t::UINT16}, {NPY_INT16,type_id_t::INT16},
      {NPY_UINT32,type_id_t::UINT32}, {NPY_INT32,type_id_t::INT32},
      {NPY_UINT64,type_id_t::UINT64}, {NPY_INT64,type_id_t::INT64},
      {NPY_FLOAT,type_id_t::FLOAT32}, {NPY_DOUBLE,type_id_t::FLOAT64},
      {NPY_LONGDOUBLE,type_id_t::FLOAT128},
      {NPY_CFLOAT,type_id_t::COMPLEX32},
      {NPY_CDOUBLE,type_id_t::COMPLEX64},
      {NPY_CLONGDOUBLE,type_id_t::COMPLEX128},
#if PY_MAJOR_VERSION >= 3
      {NPY_UNICODE,type_id_t::STRING},
#else
      {NPY_STRING,type_id_t::STRING},
#endif
      {NPY_BOOL,type_id_t::BOOL}
  };
}

namespace numpy {

ArrayAdapter::ArrayAdapter():
    pointer_(nullptr)
{}

ArrayAdapter::ArrayAdapter(const boost::python::object &object):
    pointer_(nullptr)
{
  if(!is_array(object))
  {
    throw std::runtime_error("Object is not a numpy array");
  }
  pointer_ = (PyArrayObject*)object.ptr();
}

pni::core::type_id_t ArrayAdapter::type_id() const
{
  return to_pnicore_type_id(PyArray_TYPE(pointer_));
}

hdf5::Dimensions ArrayAdapter::dimensions() const
{
  hdf5::Dimensions dims(PyArray_NDIM(pointer_));

  size_t index=0;
  for(auto &dim: dims)
    dim = (size_t)PyArray_DIM(pointer_,index++);

  return dims;
}

void *ArrayAdapter::data()
{
  return (void*)PyArray_DATA(pointer_);
}

const void *ArrayAdapter::data() const
{
  return (const void*)PyArray_DATA(pointer_);
}

size_t ArrayAdapter::size() const
{
  return PyArray_SIZE(pointer_);
}


std::vector<std::string> to_string_vector(const ArrayAdapter &array)
{
  using namespace boost::python;
  std::vector<std::string> data(array.size());

  //auto *array_ptr = static_cast<PyArrayObject*>(array);
  PyObject *ptr = PyArray_Flatten(array,NPY_CORDER);
  handle<> h(PyArray_ToList(reinterpret_cast<PyArrayObject*>(ptr)));
  list l(h);

  size_t index=0;
  for(auto &s: data) s = extract<std::string>(l[index++]);

  return data;
}


boost::python::object ArrayFactory::create(pni::core::type_id_t tid,
                                           const hdf5::Dimensions &dimensions,
                                           int itemsize)
{
  PyObject *ptr = nullptr;
  //create the buffer for with the shape information
  std::vector<npy_intp> dims(dimensions.size());
  std::copy(dimensions.begin(),dimensions.end(),dims.begin());

  ptr = reinterpret_cast<PyObject*>(
          PyArray_New(&PyArray_Type,
                       dims.size(),
                       dims.data(),
                       to_numpy_type_id(tid),
                       nullptr,
                       nullptr,
                       itemsize,
                       NPY_CORDER,
                       nullptr));


  boost::python::handle<> h(ptr);

  return boost::python::object(h);
}

boost::python::object ArrayFactory::create(const boost::python::list &list,
                                           pni::core::type_id_t tid,
                                           const hdf5::Dimensions &dimensions)
{
  //
  // create a numpy array from the list of strings
  //
  std::vector<npy_intp> dims(dimensions.size());
  std::copy(dimensions.begin(),dimensions.end(),dims.begin());

  PyArray_Dims d;
  d.ptr = dims.data();
  d.len = dims.size();
  PyObject *orig_ptr = PyArray_ContiguousFromAny(list.ptr(),to_numpy_type_id(tid),1,2);

  //
  // The resulting numpy array has the wrong shape - we fix this here
  //
  PyObject *ptr = PyArray_Newshape(reinterpret_cast<PyArrayObject*>(orig_ptr),&d,NPY_CORDER);
  boost::python::handle<> h(ptr);
  Py_XDECREF(orig_ptr);
  return boost::python::object(h);
}

int to_numpy_type_id(pni::core::type_id_t tid)
{
  return type_id2numpy_id.at(tid);
}

pni::core::type_id_t to_pnicore_type_id(int numpy_id)
{
  return numpy_id2type_id.at(numpy_id);
}

//------------------------------------------------------------------------
boost::python::object to_numpy_array(const boost::python::object &o)
{
    using namespace boost::python;
    using namespace pni::core;

    PyObject *ptr = PyArray_FROM_OF(o.ptr(),NPY_ARRAY_C_CONTIGUOUS);
    handle<> h(ptr);
    return object(h);
}

//------------------------------------------------------------------------
bool is_array(const boost::python::object &o)
{
    //if the object is not allocated we assume that it is not an array
    if(o.ptr())
        return PyArray_CheckExact(o.ptr());
    else
        return false;
}

//------------------------------------------------------------------------
bool is_scalar(const boost::python::object &o)
{
    if(o.ptr())
        return PyArray_CheckScalar(o.ptr());
    else
        return false;
}


} // namespace numpy
