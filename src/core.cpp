//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pnicore.
//
// python-pnicore is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pnicore is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pnicore.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
//  Created on: Oct 21, 2014
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <boost/python.hpp>

using namespace boost::python;

#include "bool_converter.hpp"
#include "numpy_scalar_converter.hpp"

extern void exception_registration();

//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(core)
{
    
    //this is absolutely necessary - otherwise the nympy API functions do not
    //work.
    import_array();

    //register converter
    bool_t_to_python_converter();
    python_to_bool_t_converter();
    numpy_scalar_converter();

    //register exception translators
    exception_registration();
}
