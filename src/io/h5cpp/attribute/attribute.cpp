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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

//#include "hdf5_numpy.hpp"
#include <h5cpp/hdf5.hpp>
#include <boost/python.hpp>
#include "../errors.hpp"

#if PY_MAJOR_VERSION >= 3
int
#else
void
#endif
init_numpy()
{
    import_array();
}


//
//static void read(const hdf5::attribute::Attribute &self,
//                 boost::python::object &numpy_array)
//{
//
//}
//
//static void write(const hdf5::attribute::Attribute &self,
//                  const boost::python::object &numy_array)
//{
//
//}

hdf5::attribute::Attribute
get_attribute_by_index(const hdf5::attribute::AttributeManager &self,
                        size_t index)
{
  if(index>=self.size())
    throw IndexError();

  return self[index];
}

using namespace boost::python;


BOOST_PYTHON_MODULE(_attribute)
{

    using namespace hdf5::attribute;

    init_numpy();

    //
    // setting up the documentation options
    //
    docstring_options doc_opts;
    doc_opts.disable_signatures();
    doc_opts.enable_user_defined();

    Attribute (AttributeManager::*get_by_name)(const std::string &) const = &AttributeManager::operator[];
    Attribute (AttributeManager::*get_by_index)(size_t) const = &AttributeManager::operator[];
    void (AttributeManager::*remove_by_name)(const std::string &) const = &AttributeManager::remove;
    void (AttributeManager::*remove_by_index)(size_t) const = &AttributeManager::remove;

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif
    class_<AttributeManager>("AttributeManager",init<const AttributeManager&>())
        .add_property("size",&AttributeManager::size)
        .def("__getitem__",get_by_name)
        .def("__getitem__",get_attribute_by_index)
        .def("__len__",&AttributeManager::size)
//         .def("create",&AttributeManagerWrapper::create,("name","type",arg("shape")=list(),
//             arg("overwrite")=false))
        .def("remove",remove_by_name)
        .def("remove",remove_by_index)
        .def("exists",&AttributeManager::exists)
        ;
#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif

    class_<Attribute>("Attribute")
        .add_property("datatype",&Attribute::datatype)
        .add_property("dataspace",&Attribute::dataspace)
        .add_property("name",&Attribute::name)
        .add_property("is_valid",&Attribute::is_valid)
        .add_property("parent_link",make_function(&Attribute::parent_link,return_internal_reference<>()))
        .def("close",&Attribute::close)
//        .def("read",read)
//        .def("write",write)
        ;

}

