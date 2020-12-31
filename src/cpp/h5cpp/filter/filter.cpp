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
// Created on: Jan 31, 2018
//     Authors:
//             Eugen Wintersberger <eugen.wintersberger@desy.de>
//             Jan Kotanski <jan.kotanski@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>


using namespace boost::python;
using namespace hdf5::filter;

class DLL_EXPORT ExternalFilterWrapper : public Filter
{
 public:
  ExternalFilterWrapper(FilterID id, boost::python::list cd_values):
    Filter(id)
    {
      for (boost::python::ssize_t i = 0, end = len(cd_values); i < end; ++i){
	object o = cd_values[i];
	extract<unsigned int> s(o);
	if (s.check()){
	  cd_values_.push_back(s());
	}
      }
    }

  ExternalFilterWrapper() = delete;
  ~ExternalFilterWrapper(){}

    virtual void operator()(const hdf5::property::DatasetCreationList &dcpl,
                            Availability flag=Availability::MANDATORY) const
    {
      if(H5Pset_filter(static_cast<hid_t>(dcpl), id(), static_cast<hid_t>(flag),
		       cd_values_.size(), cd_values_.data()) < 0)
	{
	  hdf5::error::Singleton::instance().throw_with_stack("Could not apply external filter!");
	}
    }

    const boost::python::list cd_values() const noexcept
    {
      boost::python::list cdlist;
      for (auto cd: cd_values_){
	cdlist.append(cd);
      }
      return cdlist;
    }

 private:
    std::vector<unsigned int> cd_values_;

};

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


BOOST_PYTHON_MODULE(_filter)
{
  using namespace boost::python;
  using namespace hdf5::filter;

  enum_<Availability>("Availability")
      .value("MANDATORY",Availability::MANDATORY)
      .value("OPTIONAL",Availability::OPTIONAL);

  class_<Filter,boost::noncopyable>("Filter",no_init)
      .add_property("id",&Filter::id)
      .def("__call__",&Filter::operator(),(args("dcpl"),args("availability")=Availability::MANDATORY))
      .def("is_encoding_enabled", &Filter::is_encoding_enabled)
      .def("is_decoding_enabled", &Filter::is_decoding_enabled)
          ;

  class_<Fletcher32,bases<Filter>>("Fletcher32");

  void (Deflate::*set_level)(unsigned int) = &Deflate::level;
  unsigned int(Deflate::*get_level)() const = &Deflate::level;
  class_<Deflate,bases<Filter>>("Deflate")
      .def(init<unsigned int>((arg("level")=0)))
      .add_property("level",get_level,set_level)
          ;

  class_<Shuffle,bases<Filter>>("Shuffle");

  const boost::python::list(ExternalFilterWrapper::*cd_values)() const = &ExternalFilterWrapper::cd_values;
  class_<ExternalFilterWrapper, bases<Filter>, boost::noncopyable>("ExternalFilter",no_init)
    .def(init<unsigned int, boost::python::list>((arg("id"), args("cd_values"))))
    .add_property("cd_values", cd_values)
    ;

  def("is_filter_available", is_filter_available, args("id"));
}
