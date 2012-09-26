
#ifndef __UTILITYFUNCTIONTEST_HPP__
#define __UTILITYFUNCTIONTEST_HPP__

#include <boost/current_function.hpp>

#include <list>
#include <vector>
#include <pni/utils/DBuffer.hpp>
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
    CPPUNIT_TEST(test_list2container<CREATE_CTYPE(std::list)>);
    CPPUNIT_TEST(test_list2container<CREATE_CTYPE(DBuffer)>);
    CPPUNIT_TEST(test_container2list<CREATE_CTYPE(std::vector)>);
    CPPUNIT_TEST(test_container2list<CREATE_CTYPE(std::list)>);
    CPPUNIT_TEST(test_container2list<CREATE_CTYPE(DBuffer)>);
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

    //convert the list to the C++ container
    auto c = List2Container<CTYPE>(l);

    //check container content
    size_t index=0;
    for(auto v: c) 
        CPPUNIT_ASSERT(v==String(extract<String>(l[index++])));

    
}

//-----------------------------------------------------------------------------
template<typename CTYPE> void UtilityFunctionTest::test_container2list()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;
    
    //create container
    CTYPE c(10);
    std::stringstream ss;
    size_t index=0;
    for(typename CTYPE::value_type &v: c) 
    {
        ss<<index++;
        v = ss.str();
    }

    list l=Container2List(c);
    index=0;
    for(auto v: c)
        CPPUNIT_ASSERT(v==String(extract<String>(l[index++])));
     
}
#endif
