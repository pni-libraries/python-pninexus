//
// (c) Copyright 2018 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// along with pyton-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 23, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "nxattribute_manager_wrapper.hpp"

AttributeManagerWrapper::AttributeManagerWrapper(const hdf5::attribute::AttributeManager &manager):
  manager_(manager)
{ }

bool AttributeManagerWrapper::exists(const pni::core::string &name) const
{
    return manager_.exists(name);
}

void AttributeManagerWrapper::remove(const pni::core::string &name) const
{
    manager_.remove(name);
}

size_t AttributeManagerWrapper::size() const
{
    return manager_.size();
}

hdf5::attribute::Attribute AttributeManagerWrapper::get_by_name(const pni::core::string &name)
{
    return manager_[name];
}

//--------------------------------------------------------------------
hdf5::attribute::Attribute AttributeManagerWrapper::get_by_index(size_t i)
{
    return manager_[i];
}

//--------------------------------------------------------------------
 hdf5::attribute::Attribute AttributeManagerWrapper::create(const pni::core::string &name,
                       const pni::core::string &type,
                       const boost::python::object &shape,
                       bool overwrite)
 {
     using namespace pni::core;
     using namespace boost::python;

     //
     // first we check if the attribute already exists and remove it if
     // requested by the user
     //
     if(manager_.exists(name) && overwrite)
       manager_.remove(name);

     //
     // create the datatype
     //
     hdf5::datatype::Datatype data_type = pni::io::nexus::DatatypeFactory::create(type_id_from_str(type));

     //
     // obtain the dimensions passed by the user
     //
     auto dimensions = Tuple2Container<hdf5::Dimensions>(tuple(shape));

     if(dimensions.empty())
     {
       //create a scalar attribute
       return manager_.create(name,data_type,hdf5::dataspace::Scalar());
     }
     else
     {
       return manager_.create(name,data_type,hdf5::dataspace::Simple(dimensions));
     }
 }

 boost::python::object AttributeManagerWrapper::__iter__()
 {
   //we return by value here and thus create a new object anyhow
   return boost::python::object(nexus::AttributeIteratorWrapper(manager_.begin(),manager_.end()));
 }

 void wrap_nxattribute_manager(const char *class_name)
 {
     using namespace boost::python;

 #ifdef __GNUG__
 #pragma GCC diagnostic push
 #pragma GCC diagnostic ignored "-Wunused-value"
 #endif
     class_<AttributeManagerWrapper>(class_name,init<const hdf5::attribute::AttributeManager&>())
         .add_property("size",&AttributeManagerWrapper::size)
         .def("__getitem__",&AttributeManagerWrapper::get_by_name)
         .def("__getitem__",&AttributeManagerWrapper::get_by_index)
         .def("__len__",&AttributeManagerWrapper::size)
         .def("__iter__",&AttributeManagerWrapper::__iter__)
         .def("create",&AttributeManagerWrapper::create,("name","type",arg("shape")=list(),
                      arg("overwrite")=false))
         .def("remove",&AttributeManagerWrapper::remove)
         .def("exists",&AttributeManagerWrapper::exists)
         ;
 #ifdef __GNUG__
 #pragma GCC diagnostic pop
 #endif
 }
