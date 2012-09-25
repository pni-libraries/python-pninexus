
#ifndef __UTILITYFUNCTIONTEST_HPP__
#define __UTILITYFUNCTIONTEST_HPP__

#include <boost/current_function.hpp>

#include <pni/nx/NX.hpp>
#include "../src/NXWrapperHelpers.hpp"

#include<cppunit/TestFixture.h>
#include<cppunit/extensions/HelperMacros.h>

using namespace pni::utils;
using namespace pni::nx::h5;

class UtilityFunctionTest:public CppUnit::TestFixture 
{
	CPPUNIT_TEST_SUITE(UtilityFunctionTest);
	CPPUNIT_TEST(test_list2container);
    CPPUNIT_TEST(test_container2list);
	CPPUNIT_TEST_SUITE_END();
public:
	void setUp();
	void tearDown();

    void test_list2container();
    void test_container2list();

};

#endif
