//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Aug 12, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <pni/core/types.hpp>
#include <pni/io/nx/nxpath.hpp>

class nxpath_element_wrapper
{
    private:
        nxpath::element_type _element;
    public:

        string get_name() const 
        { 
            return _element.first; 
        }

        void set_name(const string &name) 
        {
            _element.first = name;
        }

        string get_base_class() const
        {
            return _element.second;
        }

        void set_base_class(const string &base_class)
        {
            _element.second = base_class;
        }

};

void nxpath_element_wrapper()
{
    typedef nxpath::element_type
    class_<nxpath::element_type>("nxpath_element")
        .def(init<>())
        .add_property(

}


