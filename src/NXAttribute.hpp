#ifndef __NXATTRIBUTEPY_HPP__
#define __NXATTRIBUTEPY_HPP__

#include <pni/utils/Types.hpp>
#include <pni/utils/Shape.hpp>
#include <pni/utils/Scalar.hpp>
#include <pni/utils/Array.hpp>
#include <pni/utils/Buffer.hpp>

#include <pni/nx/NX.hpp>

using namespace pni::utils;

template<typename AttributeClass> 
class AttributeWrapper:private AttributeClass
{
    public:
        //! conversion copy constructor

        //! Creates an AttributeWrapper object  from an AttributeClass object.
        AttributeWrapper(const AttributeClass &c):
            AttributeClass(c)
        {
        }

        //! conversion move constructor

        //! Creates an AttributeWrapper object form an AttributeClass object.
        AttributeWrapper(AttributeClass &&c):
            AttributeClass(std::move(c))
        {
        }

        //! conversion copy assignment
        AttributeWrapper &operator=(const AttributeClass &c)
        {
            AttributeClass::operator=(c);
            return *this;
        }

        //! conversion move assignment
        AttributeWrapper &operator=(AttributeClass &&c)
        {
            AttributeClass::operator=(std::move(c));
            return *this;
        }

        Shape shape() const
        {
            return AttributeClass::shape();
        }

        TypID type_id() const
        {
            return AttributeClass::type_id();
        }

        void close()
        {
            AttributeClass::close();
        }
        
};


#endif
