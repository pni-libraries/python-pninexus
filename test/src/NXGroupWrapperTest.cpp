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
    group_wrapper_t g;
    CPPUNIT_ASSERT(!g.is_valid());

    g = _file.create_group("test");
    CPPUNIT_ASSERT(g.is_valid());
    CPPUNIT_ASSERT(g.name()=="test");
    CPPUNIT_ASSERT(g.base() == "/");
    CPPUNIT_ASSERT(g.path() == "/test");

    g.close();
    CPPUNIT_ASSERT(!g.is_valid());
}

//-----------------------------------------------------------------------------
void NXGroupWrapperTest::test_open()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

    _file.create_group("/test/detector");
    CPPUNIT_ASSERT(_file.exists("test"));
    CPPUNIT_ASSERT(_file.exists("test/detector"));
  
    object o = _file.open_by_name("test");
    std::cout<<"finished"<<std::endl;

}

//-----------------------------------------------------------------------------
void NXGroupWrapperTest::test_attributes()
{
    group_wrapper_t g = _file.create_group("test");
    NXObjectWrapperTest::test_attributes(g);
}
