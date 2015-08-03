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
//  Created on: Fri 24, 2015
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <vector>
#include <list>

#include "../src/utils.hpp"


using namespace boost::python; 


//-----------------------------------------------------------------------------
object list_from_vector()
{
    std::vector<int> data{1,2,3,4};

    return Container2List(data);
}

//-----------------------------------------------------------------------------
object list_from_list()
{
    std::list<int> data{1,2,3,4};

    return Container2List(data);
}

//-----------------------------------------------------------------------------
bool vector_from_list(const object &l)
{
    typedef std::vector<int> container_type; 

    auto data = List2Container<container_type>(list(l));

    if(data.size()!=4) return false;

    int index=0;
    for(auto d: data)
        if(d!=index++) return false;

    return true;
}

//-----------------------------------------------------------------------------
BOOST_PYTHON_MODULE(utils_test)
{
    def("list_from_vector",list_from_vector);
    def("list_from_list",list_from_list);
    def("vector_from_list",vector_from_list);
    def("is_unicode",is_unicode);
    def("unicode2str",unicode2str);
    def("is_int",is_int);
    def("is_bool",is_bool);
    def("is_long",is_long);
    def("is_float",is_float);
    def("is_complex",is_complex);
    def("is_string",is_string);
    def("is_scalar",is_scalar);

}

