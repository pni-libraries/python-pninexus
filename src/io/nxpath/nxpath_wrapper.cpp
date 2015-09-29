//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pnicore.
//
// python-pnicore is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pnicore is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pnicore.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Aug 12, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <pni/core/types.hpp>
#include <pni/io/nx/nxpath.hpp>
#include "../errors.hpp"

using namespace pni::core;
using namespace pni::io::nx;
using namespace boost::python;

class nxpath_iterator
{
    public:
        typedef nxpath::const_iterator iterator_type;
    private:
        iterator_type _begin;
        iterator_type _end;
    public:
        nxpath_iterator():
            _begin(),
            _end()
        {}

        nxpath_iterator(const iterator_type &b,
                        const iterator_type &e):
            _begin(b),
            _end(e)
        {}

        void increment()
        {
            _begin++;
        }

        object __iter__() const
        {
            return object(nxpath_iterator(_begin,_end));
        }

        object next() 
        {
            if(_begin==_end)
            {
                throw(nxpath_iterator_stop());
                return object();
            }
            
            auto o = *_begin;
            increment();
            return object(o);
        }
};

nxpath_iterator get_iterator(const nxpath &p)
{
    return nxpath_iterator(p.begin(),p.end());
}


void wrap_nxpath()
{
    //------------------iterator wrapper--------------------------------------
    class_<nxpath_iterator>("nxpath_iterator")
        .def("increment",&nxpath_iterator::increment)
        .def("__iter__",&nxpath_iterator::__iter__)
#if PY_MAJOR_VERSION >= 3
        .def("__next__",&nxpath_iterator::next);
#else
        .def("next",&nxpath_iterator::next);
#endif

    //-------------------nxpath wrapper---------------------------------------
    void (nxpath::*set_filename)(const string &) = &nxpath::filename;
    string (nxpath::*get_filename)()const = &nxpath::filename;
    void (nxpath::*set_attribute)(const string &) = &nxpath::attribute;
    string (nxpath::*get_attribute)() const = &nxpath::attribute;
    class_<nxpath>("nxpath")
        .add_property("front",&nxpath::front)
        .add_property("back",&nxpath::back)
        .add_property("size",&nxpath::size)
        .add_property("filename",get_filename,set_filename)
        .add_property("attribute",get_attribute,set_attribute)
        .def("_push_back",&nxpath::push_back)
        .def("_push_front",&nxpath::push_front)
        .def("_pop_back",&nxpath::pop_back)
        .def("_pop_front",&nxpath::pop_front)
        .def("__str__",&nxpath::to_string)
        .def("__len__",&nxpath::size)
        .def("__iter__",&get_iterator);


    def("make_path",nxpath::from_string);

    bool (*nxpath_match_str_str)(const string &,const string &) = &match;
    bool (*nxpath_match_path_path)(const nxpath &,const nxpath &) = &match;
    def("match",nxpath_match_str_str);
    def("match",nxpath_match_path_path);
    def("join",&join);

    def("is_root_element",&is_root_element);
    def("is_absolute",&is_absolute);
    def("is_empty",&is_empty);
    def("has_name",&has_name);
    def("has_class",&has_class);

}

