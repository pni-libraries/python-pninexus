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

#include "attribute_manager_wrapper.hpp"
#include "iterator_wrapper.hpp"

 using namespace hdf5::attribute;




////--------------------------------------------------------------------
// hdf5::attribute::Attribute AttributeManagerWrapper::create(const pni::core::string &name,
//                       const pni::core::string &type,
//                       const boost::python::object &shape,
//                       bool overwrite)
// {
//     using namespace pni::core;
//     using namespace boost::python;
//
//     //
//     // first we check if the attribute already exists and remove it if
//     // requested by the user
//     //
//     if(manager_.exists(name) && overwrite)
//       manager_.remove(name);
//
//     //
//     // create the datatype
//     //
//     hdf5::datatype::Datatype data_type = pni::io::nexus::DatatypeFactory::create(type_id_from_str(type));
//
//     //
//     // obtain the dimensions passed by the user
//     //
//     auto dimensions = Tuple2Container<hdf5::Dimensions>(tuple(shape));
//
//     if(dimensions.empty())
//     {
//       //create a scalar attribute
//       return manager_.create(name,data_type,hdf5::dataspace::Scalar());
//     }
//     else
//     {
//       return manager_.create(name,data_type,hdf5::dataspace::Simple(dimensions));
//     }
// }

 boost::python::object __iter__(const AttributeManager &self)
 {
   using namespace boost::python;
   //we return by value here and thus create a new object anyhow
   return object(AttributeIteratorWrapper(self.begin(),self.end()));
 }


 void wrap_attribute_manager(const char *class_name)
 {
     using namespace boost::python;


     Attribute (AttributeManager::*get_by_name)(const std::string &) const = &AttributeManager::operator[];
     Attribute (AttributeManager::*get_by_index)(size_t) const = &AttributeManager::operator[];
     void (AttributeManager::*remove_by_name)(const std::string &) const = &AttributeManager::remove;
     void (AttributeManager::*remove_by_index)(size_t) const = &AttributeManager::remove;

 #ifdef __GNUG__
 #pragma GCC diagnostic push
 #pragma GCC diagnostic ignored "-Wunused-value"
 #endif
     class_<AttributeManager>(class_name,init<const AttributeManager&>())
         .add_property("size",&AttributeManager::size)
         .def("__getitem__",get_by_name)
         .def("__getitem__",get_by_index)
         .def("__len__",&AttributeManager::size)
         .def("__iter__",__iter__)
//         .def("create",&AttributeManagerWrapper::create,("name","type",arg("shape")=list(),
//             arg("overwrite")=false))
         .def("remove",remove_by_name)
         .def("remove",remove_by_index)
         .def("exists",&AttributeManager::exists)
         ;
 #ifdef __GNUG__
 #pragma GCC diagnostic pop
 #endif
 }
