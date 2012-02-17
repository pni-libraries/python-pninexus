/*
 * NXField.cpp
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
#include "../src/NX.hpp"

using namespace pni::utils;
using namespace pni::nx::h5;
using namespace boost::python;

void wrap_nxfield(){
    class_<NXField,bases<NXObject> >("NXField")
        .def(init<>())
        .add_property("shape",make_function(&NXField::shape,return_internal_reference<1>()))
        .add_property("element_shape",make_function(&NXField::element_shape,return_internal_reference<1>()))
        .add_property("type_id",&NXField::type_id)
        .def("close",&NXField::close)
        ;
}
