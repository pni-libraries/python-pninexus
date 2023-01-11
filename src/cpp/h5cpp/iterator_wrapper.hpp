//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Aug 12, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include "errors.hpp"
    
template<typename IterT>
class IteratorWrapper
{
  public:
    using IteratorType = IterT;
  private:

    IteratorType begin_;
    IteratorType end_;

  public:

    IteratorWrapper(const IteratorType &b,
                    const IteratorType &e):
                      begin_(b),
                      end_(e)
  {}

    void increment()
    {
      begin_++;
    }

    boost::python::object __iter__() const
    {
      return boost::python::object(IteratorWrapper(begin_,end_));
    }

    boost::python::object next()
    {
      if(begin_==end_)
      {
        throw(StopIteration());
        return boost::python::object();
      }

      auto o = *begin_;
      increment();
      return boost::python::object(o);
    }
};
    

using NodeIteratorWrapper = IteratorWrapper<hdf5::node::NodeIterator>;
using RecursiveNodeIteratorWrapper = IteratorWrapper<hdf5::node::RecursiveNodeIterator>;
using LinkIteratorWrapper = IteratorWrapper<hdf5::node::LinkIterator>;
using RecursiveLinkIteratorWrapper = IteratorWrapper<hdf5::node::RecursiveLinkIterator>;
using AttributeIteratorWrapper = IteratorWrapper<hdf5::attribute::AttributeIterator>;
    
template<typename WrapperT>
void wrap_iterator(const char *class_name)
{
  using namespace boost::python;
  docstring_options doc_options(true,true);
  using Iterator = typename WrapperT::IteratorType;

  //------------------iterator wrapper--------------------------------------
  class_<WrapperT>(class_name,init<Iterator,Iterator>())
            .def("increment",&WrapperT::increment)
            .def("__iter__",&WrapperT::__iter__)
#if PY_MAJOR_VERSION >= 3
            .def("__next__",&WrapperT::next);
#else
            .def("next",&WrapperT::next);
#endif

}
    
