/*
 * Declaration of Nexus object template.
 *
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of libpninx.
 *
 * libpninxpython is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * libpninxpython is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libpninxpython.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************/

extern "C"
{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include<iostream>
#include<string>
#include<vector>

#include<cppunit/extensions/HelperMacros.h>
#include<cppunit/TestCaller.h>
#include<cppunit/TestResult.h>
#include<cppunit/TestRunner.h>
#include<cppunit/TextTestProgressListener.h>
#include<cppunit/ui/text/TextTestRunner.h>
#include<cppunit/extensions/TestFactoryRegistry.h>

#include<boost/python.hpp>


int main(int argc,char **argv)
{

    //need to start python - this is important as all other python functions
    //wont work if Python is not loaded.
    Py_Initialize();

    _import_array();
    //setup the test runner
    CppUnit::TextTestRunner runner;
    CppUnit::TextTestProgressListener progress;
    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
    
    runner.addTest(registry.makeTest());
    runner.eventManager().addListener(&progress);
    
    runner.run();


    return 0;
}
