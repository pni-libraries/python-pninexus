//
// (c) Copyright 2018 DESY
//
// This file is part of python-pninexus.
//
// python-pninexus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pninexus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 26, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>

void create_chunk_cache_parameters_wrapper()
{
  using namespace boost::python;
  using namespace hdf5::property;

  size_t (ChunkCacheParameters::*get_chunk_slots)() const = &ChunkCacheParameters::chunk_slots;
  void (ChunkCacheParameters::*set_chunk_slots)(size_t) = &ChunkCacheParameters::chunk_slots;
  size_t (ChunkCacheParameters::*get_chunk_cache_size)() const = &ChunkCacheParameters::chunk_cache_size;
  void (ChunkCacheParameters::*set_chunk_cache_size)(size_t) = &ChunkCacheParameters::chunk_cache_size;
  double (ChunkCacheParameters::*get_preemption_policy)() const = &ChunkCacheParameters::preemption_policy;
  void (ChunkCacheParameters::*set_preemption_policy)(double) = &ChunkCacheParameters::preemption_policy;
  class_<ChunkCacheParameters>("ChunkCacheParameters")
      .def(init<>())
      .def(init<size_t,size_t,double>())
      .add_property("chunk_slots",get_chunk_slots,set_chunk_slots)
      .add_property("chunk_cache_size",get_chunk_cache_size,set_chunk_cache_size)
      .add_property("preemption_policy",get_preemption_policy,set_preemption_policy)
      ;
}
