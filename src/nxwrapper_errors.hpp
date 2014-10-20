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
//! \brief generate translator name
//!
#define ERR_TRANSLATOR_NAME(ETYPE) ETYPE ## _translator

//----------------------------------------------------------------------------
//!
//! \ingroup errors
//! \brief generate exception pointer name
//!
#define ERR_PTR_NAME(ETYPE) Py ## ETYPE ## Ptr

//----------------------------------------------------------------------------
//!
//! \ingroup errors
//! \brief generate exception object name
//! 
#define ERR_OBJ_NAME(ETYPE) Py ## ETYPE

//----------------------------------------------------------------------------
//!
//! \ingroup errors
//! \brief generate the error translator
//! 
//! This macro creates the error translator function along with the 
//! corresponding pointer to the exception object.
//!
#define ERR_TRANSLATOR(NS,ETYPE)\
    PyObject *ERR_PTR_NAME(ETYPE) = nullptr;\
    void ERR_TRANSLATOR_NAME(ETYPE)(const NS::ETYPE &error)\
    {\
        assert(ERR_PTR_NAME(ETYPE) != nullptr);\
        object exception(error);\
        PyErr_SetObject(ERR_PTR_NAME(ETYPE),exception.ptr());\
    }

//----------------------------------------------------------------------------
//! 
//! \ingroup errors
//! \brief generats the python exception wrapper
//! 
//! Generates the wrapper code for the python exception.
//!
#define ERR_OBJECT_DECL(NS,ETYPE)\
    object ERR_OBJ_NAME(ETYPE) = (\
            class_<NS::ETYPE,bases<exception> >(# ETYPE ));\
    ERR_PTR_NAME(ETYPE) = ERR_OBJ_NAME(ETYPE).ptr();

//----------------------------------------------------------------------------
//!
//! \ingroup errors
//! \brief generate registration code
//! 
//! Macro generates the code to register an exception at the module.
//!
#define ERR_REGISTRATION(NS,ETYPE)\
    register_exception_translator<NS::ETYPE>(ERR_TRANSLATOR_NAME(ETYPE)); 


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
//! \brief register exception translators
//! 
//! This function is called by the module in order to register all exception
//! translators.
//!
void exception_registration();

