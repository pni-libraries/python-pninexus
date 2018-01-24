//
// (c) Copyright 2018 DESY
//
// This file is part of python-pni.
//
// python-pni is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pni is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pni.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 23, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

//----------------------------------------------------------------------------
//!
//! \ingroup pnicore_converters
//! \brief convert nxfield instances to their wrapper type
//! 
//! Converts instances of nxfield to their nxfield_wrapper counterpart 
//! on the python side. 
//! 
//! \tparam FTYPE field type
//!
struct NodeToPython
{

    //------------------------------------------------------------------------
    //!
    //! \brief conversion method
    //! 
    //! \param v instance of bool_t
    //! \return Python boolean object
    //!
    static PyObject *convert(const hdf5::node::Node &node);
};
