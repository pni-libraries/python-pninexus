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
}

#include <vector>
#include <pni/core/types.hpp>
#include <pni/core/python/utils.hpp>

#include <boost/python/extract.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/stl_iterator.hpp>


using namespace boost::python;
using namespace pni::core;

typedef std::vector<pni::core::slice> slices_type;

typedef std::pair<shape_t,shape_t> shapes_type; 

shapes_type get_shapes(const object &s,const object &c);

//!
//! \ingroup utils
//! \brief get tuple from arguments
//! 
//! Function solves the problem of wrapping all arguments passed to a function
//! as Python objects into a tuple. 
//! 
tuple get_tuple_from_args(const object &args);

//----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief check if object is ellipsis
//!
//! Function returns true if an object refers to the Py_Ellipsis singleton.
//!
//! \param o python object
//! \return true if o referes to Py_Ellipsis, false otherwiwse
//!
bool is_ellipsis(const object &o);

//----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief get the size of an ellipsis
//!
//! Ellipsis are used in Python to denote an entire range of dimensions. 
//! This can drastically reduce the writing effort an make code even independent
//! of a particular number of dimensions of a container. 
//!
//! This function template determines the number of dimensions an ellipsis spans
//! 
template<
         typename LISTT,
         typename ATYPE
        >
size_t get_ellipsis_size(const LISTT &index,const ATYPE &mda)
{
    //sanity check - if the total number of elements in the index sequence
    //exceeds the rank of the array type something is wrong and an exception 
    //is thrown.
    if(len(index) > mda.rank())
        throw shape_mismatch_error(EXCEPTION_RECORD,
                "Index size exceeds rank of C++ array type!");

    //generate iterator for the index sequence
    stl_input_iterator<object> begin(index),end;

    //count number of non-ellipsis objects in the index sequence
    size_t n = std::count_if(begin,end,[](const object &o)
                                         { return !is_ellipsis(o); });

    //if n is equal to the total length of the sequence there is no 
    //ellipsis at all an we can return 0
    if(n == len(index)) return 0;

    return mda.rank()-n;
}

//----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief check if an object is a slice
//! 
//! Function returns true if the object is a slice instance. 
//! 
bool is_slice(const object &o);

//----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief return the positive index
//!
//! Python allows negative index values to mark an index which should be counted
//! from backward. This function returns the corresponding positive index 
//! to a possibly negative Python index.
//!
//! \param python_index the original python index value
//! \param n_elements the number of elements in this particular dimension
//! \return positive index value
//! 
ssize_t get_positive_index(ssize_t python_index,ssize_t n_elements);

//----------------------------------------------------------------------------

template<typename ATYPE>
std::vector<pni::core::slice> get_slices(const tuple& index,const ATYPE &array)
{
    std::vector<pni::core::slice> slices;
    //determine the size of a possible ellipsis
    size_t ellipsis_size = get_ellipsis_size(index,array);

    //get the shape of the input array
    auto shape = array.template shape<shape_t>();
   
    //get iterators for the index tuple
    stl_input_iterator<object> begin(index),end;

    //get an interator for the array type shape 
    auto shape_iter = shape.begin();

    //loop over the index tuple
    for(auto index_iter = begin;index_iter!=end;++index_iter,++shape_iter)
    {
        if(is_int(*index_iter)) 
        {   //handle ingeger
            extract<boost::python::ssize_t> index(*index_iter);
            slices.push_back(pni::core::slice(get_positive_index(index,*shape_iter)));
        }
        else if(is_ellipsis(*index_iter)) 
        {   //handle ellipsis
            std::generate_n(std::back_inserter(slices),ellipsis_size,
                            [&shape_iter](){ 
                            return pni::core::slice(0,*shape_iter++);});
        }
        else if(is_slice(*index_iter))
        {   //handle slice argument
            extract<boost::python::slice> slice(*index_iter);
        }
        else
            throw type_error(EXCEPTION_RECORD,
                    "Unkonwn index type!");

    }

    return slices;
}


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
