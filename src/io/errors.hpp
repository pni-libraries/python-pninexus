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
// Declearation of exception classes and translation functions.
//
// Created on: March 15, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <pni/core/types.hpp>

//----------------------------------------------------------------------------
//! 
//! \ingroup errors
//! \brief exception to stop iteration
//! 
//! This C++ exception will be translated to StopIteration which is expected 
//! by the Python interpreter when an iterator reaches the last element in 
//! the container.
//!
class ChildIteratorStop:public std::exception
{ };

//----------------------------------------------------------------------------
//! 
//! \ingroup errors  
//! \brief exception to stop iteration
//! 
//! This C++ exception will be translated to the StopIteration Python 
//! exception expected by the Python interpreter if an iterator reaches its 
//! last position.
//!
class AttributeIteratorStop:public std::exception
{ };

//----------------------------------------------------------------------------
//! 
//! \ingroup errors  
//! \brief exception to path iteration
//! 
//! This C++ exception will be translated to the StopIteration Python 
//! exception expected by the Python interpreter if an iterator reaches its 
//! last position.
//!
class nxpath_iterator_stop:public std::exception
{};

//----------------------------------------------------------------------------
//!
//! \brief exception to stop iteration
//! 
//! This C++ exception will be translated to StopIteration in Python and 
//! is used by all iterator types. 
//! 
class StopIteration : public std::exception
{};

//----------------------------------------------------------------------------
//! 
//! \ingroup errors  
//! \brief exception to stop recursive group iteration
//! 
//! This C++ exception will be translated to the StopIteration Python 
//! exception expected by the Python interpreter if an iterator reaches its 
//! last position.
//!
class rec_group_iterator_stop:public std::exception
{};

//----------------------------------------------------------------------------
//! 
//! \ingroup errors  
//! \brief register exception translators
//! 
//! This function is called by the module in order to register all exception
//! translators.
//!
void exception_registration();

