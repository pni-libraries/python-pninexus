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
// Created on: Sep 17, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <pni/io/nx/nx.hpp>
#include "nxfield_wrapper.hpp"
#include "nxattribute_manager_wrapper.hpp"
#include "nxgroup_wrapper.hpp"

void create_nxfield_wrappers()
{
    using namespace pni::io::nx;

    wrap_nxfield<h5::nxfield>();
    wrap_nxattribute_manager<decltype(h5::nxfield::attributes)>("nxfield_attributes");
}
