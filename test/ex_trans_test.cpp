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

#include <boost/python.hpp>
#include <pni/core/error.hpp>

using namespace boost::python;
using namespace pni::core;

void throw_memory_allocation_error()
{
    throw memory_allocation_error(EXCEPTION_RECORD,"a test case");
}

void throw_memory_not_allocated_error()
{
    throw memory_not_allocated_error(EXCEPTION_RECORD,"a test case");
}

void throw_shape_mismatch_error()
{
    throw shape_mismatch_error(EXCEPTION_RECORD,"a test case");
}

void throw_size_mismatch_error()
{
    throw size_mismatch_error(EXCEPTION_RECORD,"a test case");
}

void throw_index_error()
{
    throw index_error(EXCEPTION_RECORD,"a test case");
}

void throw_key_error()
{
    throw key_error(EXCEPTION_RECORD,"a test case");
}

void throw_file_error()
{
    throw file_error(EXCEPTION_RECORD,"a test case");
}

void throw_type_error()
{
    throw type_error(EXCEPTION_RECORD,"a test case");
}

void throw_value_error()
{
    throw value_error(EXCEPTION_RECORD,"a test case");
}

void throw_range_error()
{
    throw range_error(EXCEPTION_RECORD,"a test case");
}

void throw_not_implemented_error()
{
    throw not_implemented_error(EXCEPTION_RECORD,"a test case");
}

void throw_iterator_error()
{
    throw iterator_error(EXCEPTION_RECORD,"a test case");
}

void throw_cli_argument_error()
{
    throw cli_argument_error(EXCEPTION_RECORD,"a test case");
}

void throw_cli_error()
{
    throw cli_error(EXCEPTION_RECORD,"a test case");
}

BOOST_PYTHON_MODULE(ex_trans_test)
{
    def("throw_memory_allocation_error",throw_memory_allocation_error);
    def("throw_memory_not_allocated_error",throw_memory_not_allocated_error);
    def("throw_shape_mismatch_error",throw_shape_mismatch_error);
    def("throw_size_mismatch_error",throw_size_mismatch_error);
    def("throw_index_error",throw_index_error);
    def("throw_key_error",throw_key_error);
    def("throw_file_error",throw_file_error);
    def("throw_type_error",throw_type_error);
    def("throw_value_error",throw_value_error);
    def("throw_range_error",throw_range_error);
    def("throw_not_implemented_error",throw_not_implemented_error);
    def("throw_iterator_error",throw_iterator_error);
    def("throw_cli_argument_error",throw_cli_argument_error);
    def("throw_cli_error",throw_cli_error);
}
