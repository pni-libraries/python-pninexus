
#ifndef __NXOBJECTWRAPPERTEST_HPP__
#define __NXOBJECTWRAPPERTEST_HPP__

#include <boost/current_function.hpp>

#include <list>
#include <vector>
#include <pni/utils/Array.hpp>
#include <pni/utils/TypeIDMap.hpp>
#include <iostream>
#include <sstream>
#include <pni/nx/NX.hpp>
#include <pni/utils/service.hpp>
#include "../src/NXWrapperHelpers.hpp"
#include "../src/NXObjectWrapper.hpp"

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <RandomNumberGenerator.hpp>
#include <EqualityCheck.hpp>

using namespace pni::utils;
using namespace pni::nx::h5;

//class provides tests for functions provided by the NXObjectWrapper class
class NXObjectWrapperTest
{
    private:
        static const shape_t _shape; //!< local shape for array attributes
        
        //---------------------------------------------------------------------
        template<typename T,typename OTYPE>
        static void test_scalar_attribute(const NXObjectWrapper<OTYPE> &o)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;
            _import_array();

            typedef typename NXObjectWrapper<OTYPE>::attribute_type attr_t;
            //generate the type code for the attribute
            TypeID tid = TypeIDMap<T>::type_id;
            String type_code = typeid2str(tid);

            //create attribute name from type a name for the attribute

            String attr_name = String("attr_") + type_code + String("_")+
                               demangle_cpp_name(typeid(T).name());
                               
            //create the attribute of type NXAttributeWrapper
            attr_t attr;
            CPPUNIT_ASSERT_NO_THROW(attr =
                    o.create_attribute(attr_name,type_code));
            CPPUNIT_ASSERT(attr.is_valid());
            CPPUNIT_ASSERT(attr.name() == attr_name);
            CPPUNIT_ASSERT(attr.type_id() == type_code);

            //create data value to write
            RandomNumberGenerator<T> rand;
            T write = rand();
            CPPUNIT_ASSERT_NO_THROW(attr.write(object(write)));

            //close the attribute 
            CPPUNIT_ASSERT_NO_THROW(attr.close());
            CPPUNIT_ASSERT(!attr.is_valid());

            //reopen the attribute to read data
            CPPUNIT_ASSERT_NO_THROW(attr = o.open_attr(attr_name));
            CPPUNIT_ASSERT(attr.is_valid());
            CPPUNIT_ASSERT(attr.name() == attr_name);
            CPPUNIT_ASSERT(attr.type_id() == type_code);

            //read data from the attribute
            T read = extract<T>(attr.read());
            check_equality(read,write);
        }

        //---------------------------------------------------------------------
        template<typename T,typename OTYPE>
        static void test_array_attribute(const NXObjectWrapper<OTYPE> &o)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;
            _import_array();

            typedef typename NXObjectWrapper<OTYPE>::attribute_type attr_t;
            //generate the type code for the attribute
            TypeID tid = TypeIDMap<T>::type_id;
            String type_code = typeid2str(tid);

            //create attribute name from type a name for the attribute

            String attr_name = String("attr_") + type_code + String("_")+
                               demangle_cpp_name(typeid(T).name());
                               
            //create the attribute of type NXAttributeWrapper
            attr_t attr;
            CPPUNIT_ASSERT_NO_THROW(attr =
                    o.create_attribute(attr_name,type_code,Container2List(_shape)));
            CPPUNIT_ASSERT(attr.is_valid());
            CPPUNIT_ASSERT(attr.name() == attr_name);
            CPPUNIT_ASSERT(attr.type_id() == type_code);
            auto ashape = Tuple2Container<shape_t>(attr.shape());
            CPPUNIT_ASSERT(std::equal(ashape.begin(),ashape.end(),_shape.begin()));

            //create data value to write
            object write = CreateNumpyArray<T>(_shape);
            auto warray = Numpy2RefArray<T>(write);
            RandomNumberGenerator<T> rand;
            fill_random(warray,rand);
            CPPUNIT_ASSERT_NO_THROW(attr.write(write));

            //close the attribute 
            CPPUNIT_ASSERT_NO_THROW(attr.close());
            CPPUNIT_ASSERT(!attr.is_valid());

            //reopen the attribute to read data
            CPPUNIT_ASSERT_NO_THROW(attr = o.open_attr(attr_name));
            CPPUNIT_ASSERT(attr.is_valid());
            CPPUNIT_ASSERT(attr.name() == attr_name);
            CPPUNIT_ASSERT(attr.type_id() == type_code);
            ashape = Tuple2Container<shape_t>(attr.shape());
            CPPUNIT_ASSERT(std::equal(ashape.begin(),ashape.end(),_shape.begin()));

            //read data from the attribute
            object read;
            CPPUNIT_ASSERT_NO_THROW(read = attr.read());
            auto rarray = Numpy2RefArray<T>(read);

            for(size_t i=0;i<rarray.size();i++)
                check_equality(rarray[i],warray[i]);

        }
    public:
        //---------------------------------------------------------------------
        template<typename OTYPE>
        static void test_validity(const NXObjectWrapper<OTYPE> &o)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;
            CPPUNIT_ASSERT(o.is_valid());
        }

        //---------------------------------------------------------------------
        template<typename OTYPE>
        static void test_name(const NXObjectWrapper<OTYPE> &o,const String &n)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

            CPPUNIT_ASSERT(o.name() == n);
        }

        //---------------------------------------------------------------------
        template<typename OTYPE>
        static void test_base(const NXObjectWrapper<OTYPE> &o,const String &b)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

            CPPUNIT_ASSERT(o.base() == b);
        }

        //---------------------------------------------------------------------
        template<typename OTYPE>
        static void test_path(const NXObjectWrapper<OTYPE> &o,const String &p)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

            CPPUNIT_ASSERT(o.path() == p);
        }

        //---------------------------------------------------------------------
        template<typename OTYPE> 
        static void test_scalar_attributes(const NXObjectWrapper<OTYPE> &o)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

            test_scalar_attribute<UInt8>(o);
            test_scalar_attribute<Int8>(o);
            test_scalar_attribute<UInt16>(o);
            test_scalar_attribute<Int16>(o);
            test_scalar_attribute<UInt32>(o);
            test_scalar_attribute<Int32>(o);
            test_scalar_attribute<UInt64>(o);
            test_scalar_attribute<Int64>(o);
            
            test_scalar_attribute<Float32>(o);
            test_scalar_attribute<Float64>(o);
            test_scalar_attribute<Float128>(o);

            test_scalar_attribute<Complex32>(o);
            test_scalar_attribute<Complex64>(o);
            test_scalar_attribute<Complex128>(o);
            
            test_scalar_attribute<String>(o);
            test_scalar_attribute<Bool>(o);

        }

        //---------------------------------------------------------------------
        template<typename OTYPE> 
        static void test_array_attributes(const NXObjectWrapper<OTYPE> &o)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

            test_array_attribute<UInt8>(o);
            test_array_attribute<Int8>(o);
            test_array_attribute<UInt16>(o);
            test_array_attribute<Int16>(o);
            test_array_attribute<UInt32>(o);
            test_array_attribute<Int32>(o);
            test_array_attribute<UInt64>(o);
            test_array_attribute<Int64>(o);
            
            test_array_attribute<Float32>(o);
            test_array_attribute<Float64>(o);
            test_array_attribute<Float128>(o);

            test_array_attribute<Complex32>(o);
            test_array_attribute<Complex64>(o);
            test_array_attribute<Complex128>(o);
           
            //we cannot test strings here as they require a nested list as a
            //container and not a numpy array.
            //test_array_attribute<String>(o);
            test_array_attribute<Bool>(o);
        }


        //---------------------------------------------------------------------
        template<typename OTYPE> 
        static void test_attributes(const NXObjectWrapper<OTYPE> &o)
        {
            std::cout<<BOOST_CURRENT_FUNCTION<<std::endl;

            test_scalar_attributes(o);
            test_array_attributes(o);
        }

};


#endif
