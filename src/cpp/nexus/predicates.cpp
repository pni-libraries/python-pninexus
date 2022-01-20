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
//  Created on: Feb 8, 2018
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include <pni/nexus.hpp>

using namespace boost::python;
using namespace pni;

void create_predicate_wrappers()
{
  class_<nexus::NodePredicate,boost::noncopyable>("NodePredicate",no_init);

  class_<nexus::IsBaseClass,bases<nexus::NodePredicate>>("IsBaseClass")
      .def(init<std::string>())
      .def("__call__",&nexus::IsBaseClass::operator());

  class_<nexus::IsData,bases<nexus::IsBaseClass>>("IsData");

  class_<nexus::IsDetector,bases<nexus::IsBaseClass>>("IsDetector");

  class_<nexus::IsEntry,bases<nexus::IsBaseClass>>("IsEntry");

  class_<nexus::IsInstrument,bases<nexus::IsBaseClass>>("IsInstrument");

  class_<nexus::IsSample,bases<nexus::IsBaseClass>>("IsSample");

  class_<nexus::IsSubentry,bases<nexus::IsBaseClass>>("IsSubentry");

  class_<nexus::IsTransformation,bases<nexus::IsBaseClass>>("IsTransformation");

  def("search",&nexus::search,(arg("base"),arg("predicate"),arg("recursive")=true));

}
