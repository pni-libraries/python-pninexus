/*
 * NXNumericField.cpp
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
#include "../src/NXNumericField.hpp"
#include "../src/NX.hpp"

using namespace pni::utils;
using namespace pni::nx::h5;
using namespace boost::python;

void wrap_nxnumericfield(){
    void (NXNumericField::*append_array)(const ArrayObject &) =
        &NXNumericField::append;
    void (NXNumericField::*append_scalar)(const ScalarObject &) =
        &NXNumericField::append;

    void (NXNumericField::*get_element_array)(const size_t &,ArrayObject &) =
        &NXNumericField::get;
    void (NXNumericField::*get_element_scalar)(const size_t &,ScalarObject &) =
        &NXNumericField::get;
    void (NXNumericField::*get_all_array)(ArrayObject &) =
        &NXNumericField::get;
    void (NXNumericField::*get_all_scalar)(ScalarObject &) =
        &NXNumericField::get;

    void (NXNumericField::*set_array)(const size_t &,const ArrayObject &) =
        &NXNumericField::set;
    void (NXNumericField::*set_scalar)(const size_t &,const ScalarObject &) =
        &NXNumericField::set;

    class_<NXNumericField,bases<NXField> >("NXNumericField")
        .def(init<>())
        .add_property("size",&NXNumericField::size)
        .def("append",append_array)
        .def("append",append_scalar)
        .def("get",get_element_array)
        .def("get",get_element_scalar)
        .def("get",get_all_array)
        .def("get",get_all_scalar)
        .def("set",set_array)
        .def("set",set_scalar)
        ;
}

