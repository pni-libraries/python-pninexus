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
// Created on: Feb 17, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <vector>
#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>
#include <pni/core/python/utils.hpp>

#include <pni/io/nx/nx.hpp>

#include <boost/python/extract.hpp>


using namespace boost::python;


//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief selection from tuple 
//! 
//! Adopts the selection of a field according to a tuple with indices and 
//! slices.  In order to succeed the tuple passed to this function must contain 
//! only indices, slices, and a single ellipsis.
//! 
//! \throws type_error if one of typle element is from an unsupported type
//! \throws index_error if more than one ellipsis is contained in the tuple or 
//! if an index exceeds the number of elements along the correpsonding field 
//! dimension.
//! \throws ShapeMissmatchError if the size of the tuple exceeds the rank of 
//! the field from which the selection should be drawn.
//! \param t tuple with indices and slices
//! \param f reference to the field for which to create the selection
//! \return vector with slices
//!
template<typename FTYPE>
std::vector<pni::core::slice> create_selection(const tuple &t,const FTYPE &field)
{
    //obtain a selection object
    std::vector<pni::core::slice> selection;

    //the number of elements in the tuple must not be equal to the 
    //rank of the field. This is due to the fact that the tuple can contain
    //one ellipsis which spans over several dimensions.

    bool has_ellipsis = false;
    size_t ellipsis_size = 0;
    if(len(t) > boost::python::ssize_t(field.rank()))
        throw shape_mismatch_error(EXCEPTION_RECORD,
                "Tuple with indices, slices, and ellipsis is "
                "longer than the rank of the field - something went wrong"
                "here");
    else
        ellipsis_size = field.rank()-(len(t)-1);

    /*this loop has tow possibilities:
    -> there is no ellipse and the rank of the field is larger than the size of
       the tuple passed. In this case an IndexError will occur. In this case we 
       know immediately that a shape error occured.

    i - runs over all dimensions of the input field. 
    j - runs over all values of the tuple
    */
    for(size_t i=0,j=0;i<field.rank();i++,j++)
    {
        //-------------------manage a single index-----------------------------
        extract<boost::python::ssize_t> index(t[j]);
        if(index.check())
        {
            if(index<0)
                selection.push_back(pni::core::slice(field.template shape<shape_t>()[i]+index));
            else
                selection.push_back(pni::core::slice(index));

            continue;
        }

        //----------------------------manage a slice---------------------------
        extract<boost::python::slice> s(t[j]);
        if(s.check())
        {
            //now we have to investigate the components of the 
            //slice
            boost::python::ssize_t start;
            extract<boost::python::ssize_t> __start(s().start());
            if(__start.check())
            {
                start = __start();
                if(start < 0) start = field.template shape<shape_t>()[i]+start;
            }
            else
                start = 0;
           
            boost::python::ssize_t step;
            extract<boost::python::ssize_t> __step(s().step());
            if(__step.check())
                step = __step();
            else
                step = 1;

            boost::python::ssize_t stop;
            extract<boost::python::ssize_t> __stop(s().stop());
            if(__stop.check())
            {
                stop = __stop();
                if(stop < 0) stop = field.template shape<shape_t>()[i]+stop;
            }
            else
                stop = field.template shape<shape_t>()[i];

            selection.push_back(pni::core::slice(start,stop,step));
            continue;
        }

        //------------------------manage an ellipse----------------------------
        //if we have arrived here the only possible object the tuple entry can
        //be is an ellipsis
        const object &o = t[j];
        //throw an exception if the object is not an ellipsis
        if(Py_Ellipsis != o.ptr())
            throw type_error(EXCEPTION_RECORD,
                            "Object must be either an index, a slice,"
                            " or an ellipsis!");
        //assume here that the object is an ellipsis - this is a bit difficult
        //to handle as we do not know over how many 
        if(!has_ellipsis)
        {
            has_ellipsis = true;
            while(i<j+ellipsis_size)
                selection.push_back(pni::core::slice(0,field.template shape<shape_t>()[i++]));

            //here we have to do some magic: as the value of i is already
            //increased to the next position simply continueing the loop would
            //add an additional increment of i. Thus we have to remove the
            //additional increment
            i--;
        }
        else
            throw index_error(EXCEPTION_RECORD,"Only one ellipsis is allowed!");
    }

    return selection;

}
