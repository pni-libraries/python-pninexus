#include "UtilityFunctionTest.hpp"


CPPUNIT_TEST_SUITE_REGISTRATION(UtilityFunctionTest);

//-----------------------------------------------------------------------------
void UtilityFunctionTest::setUp()
{}

//-----------------------------------------------------------------------------
void UtilityFunctionTest::tearDown()
{}

//-----------------------------------------------------------------------------
void UtilityFunctionTest::test_typeid2str()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

    //testing non-numeric types
    CPPUNIT_ASSERT(typeid2str(TypeID::STRING)=="string");
    CPPUNIT_ASSERT(typeid2str(TypeID::BOOL)=="bool");

    //testing integer types
    CPPUNIT_ASSERT(typeid2str(TypeID::UINT8)=="uint8");
    CPPUNIT_ASSERT(typeid2str(TypeID::INT8)=="int8");
    CPPUNIT_ASSERT(typeid2str(TypeID::UINT16)=="uint16");
    CPPUNIT_ASSERT(typeid2str(TypeID::INT16)=="int16");
    CPPUNIT_ASSERT(typeid2str(TypeID::UINT32)=="uint32");
    CPPUNIT_ASSERT(typeid2str(TypeID::INT32)=="int32");
    CPPUNIT_ASSERT(typeid2str(TypeID::UINT64)=="uint64");
    CPPUNIT_ASSERT(typeid2str(TypeID::INT64)=="int64");

    //testing floating point types
    CPPUNIT_ASSERT(typeid2str(TypeID::FLOAT32)=="float32");
    CPPUNIT_ASSERT(typeid2str(TypeID::FLOAT64)=="float64");
    CPPUNIT_ASSERT(typeid2str(TypeID::FLOAT128)=="float128");

    CPPUNIT_ASSERT(typeid2str(TypeID::COMPLEX32)=="complex64");
    CPPUNIT_ASSERT(typeid2str(TypeID::COMPLEX64)=="complex128");
    CPPUNIT_ASSERT(typeid2str(TypeID::COMPLEX128)=="complex256");


}


