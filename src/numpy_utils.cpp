//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: May 14, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "numpy_utils.hpp"

namespace numpy
{

    void init_array() 
    { 
        import_array(); 
    }

    //------------------------------------------------------------------------
    bool is_array(const object &o)
    {
        init_array();
        //if the object is not allocated we assume that it is not an array
        if(o.ptr())
            return PyArray_CheckExact(o.ptr());
        else
            return false;
    }

//end of namespace
}
