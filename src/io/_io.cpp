//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
//  Created on: Jan 5, 2012
//      Author: Eugen Wintersberger
//

extern "C"{
#include<Python.h>
}

#include <boost/python.hpp>
#include <iostream>
#include <sstream>

#include <pni/io/nx/nx.hpp>
#include <pni/io/exceptions.hpp>

using namespace pni::core;
using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni::io::nx;

#include "errors.hpp"

//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(_io)
{
    //register exception translators
    exception_registration();
   
}
