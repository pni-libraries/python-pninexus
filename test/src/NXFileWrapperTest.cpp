#include "NXFileWrapperTest.hpp"
#include "NXObjectWrapperTest.hpp"

CPPUNIT_TEST_SUITE_REGISTRATION(NXFileWrapperTest);

//-----------------------------------------------------------------------------
void NXFileWrapperTest::setUp() 
{ 
    _file = create_file<NXFile>("NXFileWrapperTest.h5",true,0);
}

//-----------------------------------------------------------------------------
void NXFileWrapperTest::tearDown() 
{ 
    _file.close();
}

//-----------------------------------------------------------------------------
void NXFileWrapperTest::test_creation()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

    //default constructed file wrapper object
    file_wrapper_t file;
    CPPUNIT_ASSERT(!file.is_valid());

    //try to create a file which is already open
    CPPUNIT_ASSERT_THROW(file =
            create_file<NXFile>("NXFileWrapperTest.h5",true,0),pni::nx::NXFileError);
    //close the existing file
    _file.close();

    //create a new file
    CPPUNIT_ASSERT_NO_THROW(file = create_file<NXFile>("NXFileWrapperTest.h5",true,0));
    CPPUNIT_ASSERT(file.is_valid());
    CPPUNIT_ASSERT(!file.is_readonly());
    CPPUNIT_ASSERT_NO_THROW(file.close());
    CPPUNIT_ASSERT(!file.is_valid());

   
    //try to recreate the file without overwrite option
    CPPUNIT_ASSERT_THROW(file =
            create_file<NXFile>("NXFileWrapperTest.h5",false,0),pni::nx::NXFileError);
    CPPUNIT_ASSERT(!file.is_valid());

}

//-----------------------------------------------------------------------------
void NXFileWrapperTest::test_open()
{
    std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

    file_wrapper_t file;
    _file.close();
    CPPUNIT_ASSERT_NO_THROW(file =
            open_file<NXFile>("NXFileWrapperTest.h5",false)); 
    CPPUNIT_ASSERT(file.is_valid());
    CPPUNIT_ASSERT(!file.is_readonly());

    file.close();

    CPPUNIT_ASSERT_NO_THROW(file =
            open_file<NXFile>("NXFileWrapperTest.h5",true));
    CPPUNIT_ASSERT(file.is_valid());
    CPPUNIT_ASSERT(file.is_readonly());
}

//-----------------------------------------------------------------------------
void NXFileWrapperTest::test_attributes()
{
    NXObjectWrapperTest::test_attributes(_file);
}

