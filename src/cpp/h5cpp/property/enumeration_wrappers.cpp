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
//     Authors:
//             Eugen Wintersberger <eugen.wintersberger@desy.de>
//             Jan Kotanski <jan.kotanski@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

void create_enumeration_wrappers()
{
  using namespace boost::python;
  using namespace hdf5::property;

  enum_<DatasetFillValueStatus>("DatasetFillValueStatus","status of the fill value setup for a dataset")
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
      .value("CHUNKED",DatasetLayout::CHUNKED)
#if H5_VERSION_GE(1,10,0)
      .value("VIRTUAL",DatasetLayout::VIRTUAL)
#endif
    ;

  enum_<LibVersion>("LibVersion")
      .value("LATEST",LibVersion::LATEST)
      .value("EARLIEST",LibVersion::EARLIEST);

  enum_<CloseDegree>("CloseDegree")
      .value("WEAK", CloseDegree::WEAK)
      .value("SEMI", CloseDegree::SEMI)
      .value("STRONG", CloseDegree::STRONG)
      .value("DEFAULT", CloseDegree::DEFAULT);

#if H5_VERSION_GE(1,10,0)
  enum_<VirtualDataView>("VirtualDataView")
      .value("FIRST_MISSING",VirtualDataView::FIRST_MISSING)
      .value("LAST_AVAILABLE",VirtualDataView::LAST_AVAILABLE);
#endif
}
