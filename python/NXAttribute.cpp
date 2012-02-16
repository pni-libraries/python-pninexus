/*
 * NXAttribute.cpp
 *
 *  Created on: Feb 16, 2012
 *      Author: Eugen Wintersberger
 */




#include <boost/python.hpp>

#include <pni/utils/Types.hpp>
#include <pni/utils/Array.hpp>
#include <pni/utils/Buffer.hpp>
#include <pni/utils/Scalar.hpp>

#include "../src/h5/H5Attribute.hpp"
#include "../src/NXAttribute.hpp"

using namespace pni::utils;
using namespace pni::nx::h5;
using namespace pni::nx;
using namespace boost::python;

#include "NXAttribute.hpp"

void wrap_nxattribute(){
    
    NXATTRIBUTE_WRAPPER(NXAttribute_NXAttribute,H5Attribute);
}

