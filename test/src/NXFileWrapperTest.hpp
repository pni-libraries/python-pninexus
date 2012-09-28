#ifndef __NXFILEWRAPPERTEST_HPP__
#define __NXFILEWRAPPERTEST_HPP__

#include <boost/current_function.hpp>

#include <list>
#include <vector>
#include <pni/utils/DBuffer.hpp>
#include <iostream>
#include <sstream>
#include <pni/nx/NX.hpp>
#include "../src/NXFileWrapper.hpp"

#include<cppunit/TestFixture.h>
#include<cppunit/extensions/HelperMacros.h>

using namespace pni::utils;
using namespace pni::nx::h5;

class NXFileWrapperTest:public CppUnit::TestFixture
{
        CPPUNIT_TEST_SUITE(NXFileWrapperTest);
        CPPUNIT_TEST(test_creation);
        CPPUNIT_TEST(test_open);
        CPPUNIT_TEST(test_attributes);
        CPPUNIT_TEST_SUITE_END();
    private:
        NXFileWrapper<NXFile> _file;
    public:
        //=============public types=====================
        typedef NXFileWrapper<NXFile> file_wrapper_t;
        void setUp();
        void tearDown();
        void test_creation();
        void test_open();
        void test_attributes();
        
};

#endif
