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
#include <h5cpp/contrib/stl/stl.hpp>

void create_enumeration_wrappers()
{
  using namespace boost::python;
  using namespace hdf5::property;

  enum_<DatasetFillValueStatus>("DatasetFillValueStatus","status of the fill value setup for a dataset")
      .value("UNDEFINED",DatasetFillValueStatus::Undefined)
      .value("DEFAULT",DatasetFillValueStatus::Default)
      .value("USER_DEFINED",DatasetFillValueStatus::UserDefined);

  enum_<DatasetFillTime>("DatasetFillTime")
      .value("IFSET",DatasetFillTime::IfSet)
      .value("ALLOC",DatasetFillTime::Alloc)
      .value("NEVER",DatasetFillTime::Never);

  enum_<DatasetAllocTime>("DatasetAllocTime")
      .value("DEFAULT",DatasetAllocTime::Default)
      .value("EARLY",DatasetAllocTime::Early)
      .value("INCR",DatasetAllocTime::Incr)
      .value("LATE",DatasetAllocTime::Late);

  enum_<DatasetLayout>("DatasetLayout")
      .value("COMPACT",DatasetLayout::Compact)
      .value("CONTIGUOUS",DatasetLayout::Contiguous)
      .value("CHUNKED",DatasetLayout::Chunked)
#if H5_VERSION_GE(1,10,0)
      .value("VIRTUAL",DatasetLayout::Virtual)
#endif
    ;

  enum_<LibVersion>("LibVersion")
      .value("LATEST",LibVersion::Latest)
      .value("EARLIEST",LibVersion::Earliest);

  enum_<CloseDegree>("CloseDegree")
      .value("WEAK", CloseDegree::Weak)
      .value("SEMI", CloseDegree::Semi)
      .value("STRONG", CloseDegree::Strong)
      .value("DEFAULT", CloseDegree::Default);

#if H5_VERSION_GE(1,10,0)
  enum_<VirtualDataView>("VirtualDataView")
      .value("FIRST_MISSING",VirtualDataView::FirstMissing)
      .value("LAST_AVAILABLE",VirtualDataView::LastAvailable);
#endif
}
