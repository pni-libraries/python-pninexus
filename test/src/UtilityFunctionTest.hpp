
#ifndef __UTILITYFUNCTIONTEST_HPP__
#define __UTILITYFUNCTIONTEST_HPP__

#include <boost/current_function.hpp>

#include <iostream>
#include <sstream>
#include <pni/nx/NX.hpp>
#include "../src/NXWrapperHelpers.hpp"

#include<cppunit/TestFixture.h>
#include<cppunit/extensions/HelperMacros.h>

using namespace pni::utils;
using namespace pni::nx::h5;

#define CREATE_CTYPE(ctype) ctype<String>

class UtilityFunctionTest:public CppUnit::TestFixture 
{
	CPPUNIT_TEST_SUITE(UtilityFunctionTest);
	CPPUNIT_TEST(test_list2container<CREATE_CTYPE(std::vector)>);
    CPPUNIT_TEST(test_container2list<CREATE_CTYPE(std::vector)>);
	CPPUNIT_TEST_SUITE_END();
public:
	void setUp();
	void tearDown();

    template<typename CTYPE> void test_list2container();
    template<typename CTYPE> void test_container2list();

};

//-----------------------------------------------------------------------------
template<typename CTYPE> void UtilityFunctionTest::test_list2container()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

    //user a list of strings for testing
    list l;
    std::stringstream ss;
    for(size_t i=0;i<10;i++)
    {
        ss<<i;
        l.append(String(ss.str()));
    }

    auto c = List2Container<CTYPE>(l);

    
}

//-----------------------------------------------------------------------------
template<typename CTYPE> void UtilityFunctionTest::test_container2list()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;
}
#endif
