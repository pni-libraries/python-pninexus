/*
 * NXObject.cpp
 *
 *  Created on: Jan 5, 2012
 *      Author: Eugen Wintersberger
 */




#include <boost/python.hpp>

#include <pni/utils/Types.hpp>
#include <pni/utils/ArrayObject.hpp>
#include <pni/utils/ScalarObject.hpp>

#include "../src/NXObject.hpp"

#include "../src/h5/H5AttributeObject.hpp"
#include "../src/h5/H5Dataset.hpp"
#include "../src/h5/H5Group.hpp"
#include "../src/h5/H5File.hpp"


#include "NXObject.hpp"

using namespace pni::utils;
using namespace pni::nx::h5;
using namespace pni::nx;
using namespace boost::python;

void wrap_nxobject(){
    NXOBJECT_WRAPPER(NXObject_NXObject,H5AttributeObject);
	//===================Wrapping NumericObject=================================
    NXOBJECT_WRAPPER(NXObject_NXGroup,H5Group);
    NXOBJECT_WRAPPER(NXObject_NXField,H5Dataset);
    NXOBJECT_WRAPPERNOCOP(NXObject_NXFile,H5File);
}


