/*
 * NXBinaryField.cpp
 *
 *  Created on: Jan 6, 2012
 *      Author: Eugen Wintersberger
 */




#include <boost/python.hpp>

#include <pni/utils/Types.hpp>
#include <pni/utils/Shape.hpp>
#include <pni/utils/Buffer.hpp>

#include "../src/NXObject.hpp"
#include "../src/NXField.hpp"
#include "../src/NXBinaryField.hpp"
#include "../src/NX.hpp"

using namespace pni::utils;
using namespace pni::nx::h5;
using namespace boost::python;

void wrap_nxbinaryfield(){
    void (NXBinaryField::*get_buffer)(const size_t &,Buffer<Binary> &) =
        &NXBinaryField::get;
    void (NXBinaryField::*get_buffer_all)(Buffer<Binary> &) = &NXBinaryField::get;
    class_<NXBinaryField,bases<NXField> >("NXBinaryField")
        .def(init<>())
        .add_property("size",&NXBinaryField::size)
        .def("append",&NXBinaryField::append)
        .def("set",&NXBinaryField::set)
        .def("get",get_buffer)
        .def("get",get_buffer_all)
        ;
}

