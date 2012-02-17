/*
 * NXStringField.cpp
 *
 *  Created on: Jan 6, 2012
 *      Author: Eugen Wintersberger
 */




#include <boost/python.hpp>

#include <pni/utils/Types.hpp>
#include <pni/utils/Shape.hpp>
#include <pni/utils/ArrayObject.hpp>
#include <pni/utils/ScalarObject.hpp>
#include <pni/utils/NumericObject.hpp>


#include "../src/NXObject.hpp"
#include "../src/NXField.hpp"
#include "../src/NXStringField.hpp"
#include "../src/NX.hpp"

using namespace pni::utils;
using namespace pni::nx::h5;
using namespace boost::python;


void wrap_nxstringfield(){
    void (NXStringField::*get1)(const size_t &,String &) = &NXStringField::get;
    String (NXStringField::*get2)(const size_t &) = &NXStringField::get;
    String (NXStringField::*get3)(const char &) = &NXStringField::get;

    class_<NXStringField,bases<NXField> >("NXStringField")
       .def(init<>())
       .add_property("size",&NXStringField::size)
       .def("append",&NXStringField::append)
       .def("get",get1)
       .def("get",get2)
       .def("get",get3)
       .def("set",&NXStringField::set)
       ;
}
