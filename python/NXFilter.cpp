/*
 * NXFilter.cpp
 *
 *  Created on: Jan 6, 2012
 *      Author: Eugen Wintersberger
 */




#include <boost/python.hpp>

#include <pni/utils/Types.hpp>
#include <pni/utils/ArrayObject.hpp>
#include <pni/utils/ScalarObject.hpp>

#include "../src/NXFilter.hpp"
#include "../src/NX.hpp"

using namespace pni::utils;
using namespace boost::python;
using namespace pni::nx::h5;


void wrap_filter(){
    class_<NXLZFFilter>("NXLZFFilter")
        .def(init<>())
        .def("setup",&NXLZFFilter::setup)
        ;

    void (NXDeflateFilter::*set_compression)(UInt32) =
        &NXDeflateFilter::compression_rate;
    UInt32 (NXDeflateFilter::*get_compression)() const = 
        &NXDeflateFilter::compression_rate;
    class_<NXDeflateFilter>("NXDeflateFilter")
        .def(init<>())
        .add_property("rate",get_compression,set_compression)
        .def("setup",&NXDeflateFilter::setup)
        ;
}
