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
// Created on: Feb 31, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//


#include <numeric>
#include "dimensions.hpp"

template<typename DimT>
typename DimT::value_type get_size(const DimT &dimensions)
{
  using value_type = typename DimT::value_type;
  return std::accumulate(dimensions.begin(),dimensions.end(),value_type(1),
                         std::multiplies<value_type>());
}

namespace numpy {

Dimensions::Dimensions(const hdf5::Dimensions &dims):
    Base(dims.begin(),dims.end())
{}

Dimensions::Dimensions(const hdf5::dataspace::Selection &selection):
    Base()
{
  using hdf5::dataspace::Hyperslab;

  const Hyperslab* slab = dynamic_cast<const Hyperslab*>(&selection);

  auto block_iter = slab->block().begin();
  auto count_iter = slab->count().begin();
  for(size_t index=0;index<slab->rank();++index)
  {
    value_type dim = *block_iter++ * (*count_iter++);
    this->push_back(dim);
  }

  if(get_size(*this)==1)
  {
    *this = Base{1};
  }
  else
  {
    //remove all 1 in the dimesions, they are not needed
    erase(std::remove_if(begin(),end(),[](value_type v){ return v==1; }));
  }
}

const Dimensions::value_type* Dimensions::dims() const
{
  return data();
}

int Dimensions::ndims() const
{
  return size();
}

Dimensions::operator hdf5::Dimensions()
{
  hdf5::Dimensions h5_dims(size());
  std::copy(begin(),end(),h5_dims.begin());
  return h5_dims;
}

} // namespace numpy
