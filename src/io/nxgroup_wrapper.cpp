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
// along with pyton-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Sep 18, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <pni/io/nx/nx.hpp>
#include "nxgroup_wrapper.hpp"
#include "child_iterator.hpp"
#include "nxattribute_manager_wrapper.hpp"

void create_nxgroup_wrappers()
{
    using namespace pni::io::nx;
    wrap_nxgroup<h5::nxgroup>();
    wrap_childiterator<nxgroup_wrapper<h5::nxgroup>>("NXGroupChildIterator");
    wrap_nxattribute_manager<decltype(h5::nxgroup::attributes)>("nxgroup_attributes");
}


