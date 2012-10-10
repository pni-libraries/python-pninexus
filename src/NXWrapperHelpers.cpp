/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of libpninx-python.
 *
 * libpninx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * libpninx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libpninx.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Definition and implementation of helper functions and classes for wrappers.
 *
 * Created on: Feb 17, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

//helper functions to create wrappers

#include <boost/python/extract.hpp>
#include <boost/python/slice.hpp>
#include "NXWrapperHelpers.hpp"
#include "NXWrapperErrors.hpp"

//-----------------------------------------------------------------------------
String typeid2str(const TypeID &tid)
{
    if(tid == TypeID::STRING) return "string";
    if(tid == TypeID::UINT8) return "uint8";
    if(tid == TypeID::INT8)  return "int8";
    if(tid == TypeID::UINT16) return "uint16";
    if(tid == TypeID::INT16)  return "int16";
    if(tid == TypeID::UINT32) return "uint32";
    if(tid == TypeID::INT32)  return "int32";
    if(tid == TypeID::UINT64) return "uint64";
    if(tid == TypeID::INT64) return "int64";

    if(tid == TypeID::FLOAT32) return "float32";
    if(tid == TypeID::FLOAT64) return "float64";
    if(tid == TypeID::FLOAT128) return "float128";

    if(tid == TypeID::COMPLEX32) return "complex64";
    if(tid == TypeID::COMPLEX64) return "complex128";
    if(tid == TypeID::COMPLEX128) return "complex256";

    if(tid == TypeID::BOOL) return "bool";

    return "none";
}


//------------------------------------------------------------------------------
std::vector<Slice> create_selection(const tuple &t,const NXField &field)
{
    //obtain a selection object
    std::vector<Slice> selection;

    //the number of elements in the tuple must not be equal to the 
    //rank of the field. This is due to the fact that the tuple can contain
    //one ellipsis which spans over several dimensions.

    bool has_ellipsis = false;
    size_t ellipsis_size = 0;
    if(len(t) > boost::python::ssize_t(field.rank()))
        throw ShapeMissmatchError(EXCEPTION_RECORD,
                "Tuple with indices, slices, and ellipsis is "
                "longer than the rank of the field - something went wrong"
                "here");
    else
        ellipsis_size = field.rank()-(len(t)-1);

    /*this loop has tow possibilities:
    -> there is no ellipse and the rank of the field is larger than the size of
       the tuple passed. In this case an IndexError will occur. In this case we 
       know immediately that a shape error occured.
    */
    for(size_t i=0,j=0;i<field.rank();i++,j++)
    {
        //manage a single index
        extract<boost::python::ssize_t> index(t[j]);

        if(index.check())
        {
            if(index<0)
                selection.push_back(Slice(field.shape<shape_t>()[i]+index));
            else
                selection.push_back(Slice(index));

            continue;
        }

        //manage a slice
        extract<slice> s(t[j]);
        if(s.check())
        {
            //now we have to investigate the components of the 
            //slice
            boost::python::ssize_t start;
            extract<boost::python::ssize_t> __start(s().start());
            if(__start.check())
            {
                start = __start();
                if(start < 0) start = field.shape<shape_t>()[i]+start;
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
                if(stop < 0) stop = field.shape<shape_t>()[i]+stop;
            }
            else
                stop = field.shape<shape_t>()[i];

            selection.push_back(Slice(start,stop,step));
            continue;
        }

        //manage an ellipse
        const object &o = t[j];
        if(Py_Ellipsis != o.ptr())
            throw TypeError(EXCEPTION_RECORD,
                            "Object must be either an index, a slice,"
                            " or an ellipsis!");
        //assume here that the object is an ellipsis - this is a bit difficult
        //to handle as we do not know over how many 
        if(!has_ellipsis)
        {
            has_ellipsis = true;
            while(i<j+ellipsis_size)
                selection.push_back(Slice(0,field.shape<shape_t>()[i++]));
        }
        else
            throw IndexError(EXCEPTION_RECORD,"Only one ellipsis is allowed!");
    }

    return selection;

}

//-----------------------------------------------------------------------------
size_t nested_list_rank(const object &o)
{
    size_t rank = 0;
    extract<list> l(o);

    //check if conversion was successful
    if(l.check())
    {
        //if the list has an element
        if(len(l))
            rank = 1 + nested_list_rank(l()[0]);
        else 
            rank = 1; //list has no element
    }
    else
        rank = 0; //object is not a list

    return rank;
}
