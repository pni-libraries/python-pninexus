//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
//  Created on: Mon 27, 2015
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <vector>
#include <list>

#include "../src/numpy_utils.hpp"
#include "../src/init_numpy.hpp"
#include "../src/utils.hpp"
#include "check_type_id_from_object.hpp"


using namespace boost::python; 


BOOST_PYTHON_MODULE(numpy_utils_test)
{
    init_numpy();
    def("is_array",numpy::is_array);
    def("is_scalar",numpy::is_scalar);

    CHECK_TYPE_ID_FROM_OBJECT_EXPOSE();
}


