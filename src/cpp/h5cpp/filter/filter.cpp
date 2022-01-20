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
// Created on: Jan 31, 2018
//     Authors:
//             Eugen Wintersberger <eugen.wintersberger@desy.de>
//             Jan Kotanski <jan.kotanski@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>


using namespace boost::python;
using namespace hdf5::filter;

class DLL_EXPORT ExternalFilterWrapper : public Filter
{
 public:
  ExternalFilterWrapper(FilterID id, boost::python::list cd_values,
			const std::string &name=std::string()):
    Filter(id)
    {
      for (boost::python::ssize_t i = 0, end = len(cd_values); i < end; ++i){
	object o = cd_values[i];
	extract<unsigned int> s(o);
	if (s.check()){
	  cd_values_.push_back(s());
	}
      }
      name_ = name;
    }
  ExternalFilterWrapper():
    Filter(0),
    cd_values_(0, 0),
    name_("")
    {
    }

  ~ExternalFilterWrapper(){}

    virtual void operator()(const hdf5::property::DatasetCreationList &dcpl,
                            Availability flag=Availability::Mandatory) const
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

    const std::string name() const noexcept
    {
      return name_;
    }

 private:
    std::vector<unsigned int> cd_values_;
    std::string name_;


};

DLL_EXPORT boost::python::list externalfilters_fill(boost::python::list &efilters,
						    const hdf5::property::DatasetCreationList &dcpl,
						    size_t max_cd_number=16,
						    size_t max_name_size=257){

  boost::python::list flags;
  size_t nfilters = dcpl.nfilters();
  unsigned int flag;
  size_t cd_number = max_cd_number;
  std::vector<char> fname(max_name_size);

  for(unsigned int nf=0; nf != nfilters; nf++){
    std::vector<unsigned int> cd_values(max_cd_number);
    int filter_id = H5Pget_filter(static_cast<hid_t>(dcpl),
				  nf,
				  &flag,
				  &cd_number,
				  cd_values.data(),
				  fname.size(),
				  fname.data(),
				  NULL);

    if(filter_id < 0){
      std::stringstream ss;
      ss << "Failure to read a parameters of filter ("
	 << nf << ") from " << dcpl.get_class();
      hdf5::error::Singleton::instance().throw_with_stack(ss.str());
    }
    if(cd_number > max_cd_number){
      std::stringstream ss;
      ss<<"Too many filter parameters in " << dcpl.get_class();
      hdf5::error::Singleton::instance().throw_with_stack(ss.str());
    }
    cd_values.resize(cd_number);
    if(static_cast<int>(static_cast<Availability>(flag)) != flag){
      std::stringstream ss;
      ss<<"Wrong filter flag value in " << dcpl.get_class();
      hdf5::error::Singleton::instance().throw_with_stack(ss.str());
    }

    boost::python::list cdlist;
    for (auto cd: cd_values)
      cdlist.append(cd);


    Availability fflag = static_cast<Availability>(flag);
    fname[max_name_size - 1] = '\0';
    std::string name(fname.data());
    efilters.append(ExternalFilterWrapper(filter_id, cdlist, name));
    flags.append(fflag);
  }
  return flags;
}

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


BOOST_PYTHON_MODULE(_filter)
{
  using namespace boost::python;
  using namespace hdf5::filter;

  enum_<Availability>("Availability")
      .value("MANDATORY",Availability::Mandatory)
      .value("OPTIONAL",Availability::Optional);
  
  enum_<ScaleOffset::ScaleType>("SOScaleType")
      .value("FLOAT_DSCALE",ScaleOffset::ScaleType::FloatDScale)
      .value("FLOAT_ESCALE",ScaleOffset::ScaleType::FloatEScale)
      .value("INT",ScaleOffset::ScaleType::Int);

  enum_<SZip::OptionMask>("SZipOptionMask")
    .value("NONE", SZip::OptionMask::None)
    .value("ALLOW_K13", SZip::OptionMask::AllowK13)
    .value("CHIP", SZip::OptionMask::Chip)
    .value("ENTROPY_CODING", SZip::OptionMask::EntropyCoding)
    .value("NEAREST_NEIGHBOR", SZip::OptionMask::NearestNeighbor);

  class_<Filter,boost::noncopyable>("Filter",no_init)
      .add_property("id",&Filter::id)
      .def("__call__",&Filter::operator(),(args("dcpl"),args("availability")=Availability::Mandatory))
      .def("is_encoding_enabled", &Filter::is_encoding_enabled)
      .def("is_decoding_enabled", &Filter::is_decoding_enabled)
          ;

  class_<Fletcher32,bases<Filter>>("Fletcher32");

  class_<NBit,bases<Filter>>("NBit");

  void (Deflate::*set_level)(unsigned int) = &Deflate::level;
  unsigned int(Deflate::*get_level)() const = &Deflate::level;
  class_<Deflate,bases<Filter>>("Deflate")
      .def(init<unsigned int>((arg("level")=0)))
      .add_property("level",get_level,set_level)
          ;
  
  void (SZip::*set_option_mask)(unsigned int) = &SZip::option_mask;
  unsigned int(SZip::*get_option_mask)() const = &SZip::option_mask;
  void (SZip::*set_pixels_per_block)(unsigned int) = &SZip::pixels_per_block;
  unsigned int(SZip::*get_pixels_per_block)() const = &SZip::pixels_per_block;
  class_<SZip,bases<Filter>>("SZip")
    .def(init<unsigned int,unsigned int>((arg("option_mask")=32,
					  arg("pixels_per_block")=0)))
    .add_property("option_mask",get_option_mask,set_option_mask)
    .add_property("pixels_per_block",get_pixels_per_block,set_pixels_per_block)
    ;

  void (ScaleOffset::*set_scale_type)(ScaleOffset::ScaleType) = &ScaleOffset::scale_type;
  ScaleOffset::ScaleType(ScaleOffset::*get_scale_type)() const = &ScaleOffset::scale_type;
  void (ScaleOffset::*set_scale_factor)(int) = &ScaleOffset::scale_factor;
  int(ScaleOffset::*get_scale_factor)() const = &ScaleOffset::scale_factor;
  class_<ScaleOffset,bases<Filter>>("ScaleOffset")
    .def(init<ScaleOffset::ScaleType,int>((arg("scale_type")=ScaleOffset::ScaleType::FloatDScale,
					  arg("scale_factor")=1)))
    .add_property("scale_type",get_scale_type,set_scale_type)
    .add_property("scale_factor",get_scale_factor,set_scale_factor)
    ;

  class_<Shuffle,bases<Filter>>("Shuffle");

  const boost::python::list(ExternalFilterWrapper::*cd_values)() const = &ExternalFilterWrapper::cd_values;
  const std::string(ExternalFilterWrapper::*name)() const = &ExternalFilterWrapper::name;
  class_<ExternalFilterWrapper, bases<Filter>>("ExternalFilter")
    .def(init<unsigned int, boost::python::list, std::string>((arg("id"), args("cd_values"), args("name")=std::string())))
    .add_property("cd_values", cd_values)
    .add_property("name", name)
    ;
  def("_externalfilters_fill",externalfilters_fill,
    (arg("efilters"), arg("dcpl"),arg("max_cd_number")=16,arg("max_name_size")=257));

  def("is_filter_available", is_filter_available, args("id"));
}
