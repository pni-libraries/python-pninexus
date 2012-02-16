/*
 * NXFile.cpp
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
#include "../src/NXGroup.hpp"
#include "../src/NXFile.hpp"
#include "../src/h5/NXFileH5Implementation.hpp"
#include "../src/h5/NXObjectH5Implementation.hpp"


using namespace pni::utils;
using namespace pni::nx::h5;
using namespace boost::python;

void wrap_nxfile(){
    String (pni::nx::NXFile<NXFileH5Implementation>::*get_filename)() 
        const = &pni::nx::NXFile<NXFileH5Implementation>::filename;
    void (pni::nx::NXFile<NXFileH5Implementation>::*set_filename)
        (const String &) = &pni::nx::NXFile<NXFileH5Implementation>::filename;

    bool (pni::nx::NXFile<NXFileH5Implementation>::*get_readonly)
        () const = &pni::nx::NXFile<NXFileH5Implementation>::read_only;
    void (pni::nx::NXFile<NXFileH5Implementation>::*set_readonly)
        (bool) = &pni::nx::NXFile<NXFileH5Implementation>::read_only;

    bool (pni::nx::NXFile<NXFileH5Implementation>::*get_overwrite)
        () const = &pni::nx::NXFile<NXFileH5Implementation>::overwrite;
    void (pni::nx::NXFile<NXFileH5Implementation>::*set_overwrite)
        (bool) = &pni::nx::NXFile<NXFileH5Implementation>::overwrite;

    size_t (pni::nx::NXFile<NXFileH5Implementation>::*get_splitsize)
        ()const = &pni::nx::NXFile<NXFileH5Implementation>::split_size;
    void (pni::nx::NXFile<NXFileH5Implementation>::*set_splitsize)
        (size_t) = &pni::nx::NXFile<NXFileH5Implementation>::split_size;

    void (pni::nx::NXFile<NXFileH5Implementation>::*file_open)
        () = &pni::nx::NXFile<NXFileH5Implementation>::open;
    pni::nx::NXObject<NXObjectH5Implementation> 
        (pni::nx::NXFile<NXFileH5Implementation>::*object_open)
        (const String &) = &pni::nx::NXFile<NXFileH5Implementation>::open;

    class_<pni::nx::NXFile<NXFileH5Implementation>,
            bases<pni::nx::NXGroup<NXFileH5Implementation> >,
            boost::noncopyable >("NXFile")
        .def(init<>())
        .add_property("filename",get_filename,set_filename)
        .add_property("read_only",get_readonly,set_readonly)
        .add_property("overwrite",get_overwrite,set_overwrite)
        .add_property("splitsize",get_splitsize,set_splitsize)
        .def("create",&pni::nx::NXFile<NXFileH5Implementation>::create)
        .def("open",file_open)
        .def("open",object_open)
        .def("close",&pni::nx::NXFile<NXFileH5Implementation>::close)
        .def("flush",&pni::nx::NXFile<NXFileH5Implementation>::flush)
        ;
}

