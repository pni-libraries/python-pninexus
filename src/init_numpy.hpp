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
//  Created on: Mon July 27, 2015
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
extern "C"{
#include <Python.h>
#include <numpy/arrayobject.h>
}

//!
//! \ingroup devel_doc
//! \brief numpy initialization function
//! 
//! This function must be called in any case before any of the numpy 
//! utility functions is called. Currently this function is called by 
//! the initialization function of the `pni.core` package. 
#if PY_MAJOR_VERSION >= 3
int
#else 
void
#endif
init_numpy();

