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
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 26, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

void create_enumeration_wrappers()
{
  using namespace boost::python;
  using namespace hdf5::property;

  enum_<DatasetFillValueStatus>("DatasetFillValueStatus")
      .value("UNDEFINED",DatasetFillValueStatus::UNDEFINED)
      .value("DEFAULT",DatasetFillValueStatus::DEFAULT)
      .value("USER_DEFINED",DatasetFillValueStatus::USER_DEFINED);

  enum_<DatasetFillTime>("DatasetFillTime")
      .value("IFSET",DatasetFillTime::IFSET)
      .value("ALLOC",DatasetFillTime::ALLOC)
      .value("NEVER",DatasetFillTime::NEVER);

  enum_<DatasetAllocTime>("DatasetAllocTime")
      .value("DEFAULT",DatasetAllocTime::DEFAULT)
      .value("EARLY",DatasetAllocTime::EARLY)
      .value("INCR",DatasetAllocTime::INCR)
      .value("LATE",DatasetAllocTime::LATE);

  enum_<DatasetLayout>("DatasetLayout")
      .value("COMPACT",DatasetLayout::COMPACT)
      .value("CONTIGUOUS",DatasetLayout::CONTIGUOUS)
      .value("CHUNKED",DatasetLayout::CHUNKED);

  enum_<LibVersion>("LibVersion")
      .value("LATEST",LibVersion::LATEST)
      .value("EARLIEST",LibVersion::EARLIEST);


}
