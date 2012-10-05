#include "NXGroupWrapperTest.hpp"
#include "NXObjectWrapperTest.hpp"

CPPUNIT_TEST_SUITE_REGISTRATION(NXGroupWrapperTest);

//-----------------------------------------------------------------------------
void NXGroupWrapperTest::setUp()
{
    _file = create_file<NXFile>("NXGroupWrapperTest.h5",true,0);
}

//-----------------------------------------------------------------------------
void NXGroupWrapperTest::tearDown()
{
    _file.close();
}

//-----------------------------------------------------------------------------
void NXGroupWrapperTest::test_creation()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

}

//-----------------------------------------------------------------------------
void NXGroupWrapperTest::test_open()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

}

//-----------------------------------------------------------------------------
void NXFileWrapperTest::test_attributes()
{
    group_wrapper_t g = _file.create_group("test");
    NXObjectWrapperTest::test_attribute(g);
}
