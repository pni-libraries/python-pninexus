import numpy
import platform

class attributes_test(object):

    def scalar_attribute_test(self,ts,parent,name,tc,value):
        """
        scalar_attribute_test(self,parent,name,tc)

        required positional arguments:
        ts ............. test suite
        parent ......... the parent object
        name ........... name of the attribute
        tc ............. typecode
        """

        #create the attribute
        attr = parent.attributes.create(name,tc)

        #test inquery properties
        ts.assertTrue(attr.dtype == tc)
        ts.assertTrue(attr.valid)
        ts.assertTrue(attr.shape == (1,))
        ts.assertTrue(attr.name == name)
            
        #test data io
        attr.value = value
        if tc=="string":
            ts.assertEqual(attr.value,value)
        else:
            ts.assertAlmostEqual(attr.value,value)

    def array_attribute_test(self,ts,parent,name,tc,shape,value):
        #create the attribute
        attr = parent.attributes.create(name,tc,shape)

        #test inquery properties
        print tc,attr.dtype
        ts.assertTrue(attr.dtype == tc)
        ts.assertTrue(attr.valid)
        ts.assertEqual(attr.shape,shape)
        ts.assertTrue(attr.name == name)
            
        #test data io
        attr.value = value
        for i in range(value.size):
            ts.assertAlmostEqual(value.flat[i],attr.value.flat[i])

    def test_scalar_attribute(self,ts,parent):

        self.scalar_attribute_test(ts,parent,"uint8_attr","uint8",100)
        self.scalar_attribute_test(ts,parent,"int8_attr","int8",-100)
        self.scalar_attribute_test(ts,parent,"uint16_attr","uint16",100)
        self.scalar_attribute_test(ts,parent,"int16_attr","int16",-100)
        self.scalar_attribute_test(ts,parent,"uint32_attr","uint32",100)
        self.scalar_attribute_test(ts,parent,"int32_attr","int32",-100)
        self.scalar_attribute_test(ts,parent,"uint64_attr","uint64",100)
        self.scalar_attribute_test(ts,parent,"int64_attr","int64",-100)

        self.scalar_attribute_test(ts,parent,"float32_attr","float32",1.234)
        self.scalar_attribute_test(ts,parent,"float64_attr","float64",-2034.13)
        self.scalar_attribute_test(ts,parent,"float128_attr","float128",-13423542.23434)

        self.scalar_attribute_test(ts,parent,"complex32_attr","complex32",1+123.j)
        self.scalar_attribute_test(ts,parent,"complex64_attr","complex64",1+123.j)
        self.scalar_attribute_test(ts,parent,"complex128_attr","complex128",1+123.j)

        self.scalar_attribute_test(ts,parent,"text","string","hello world this is a text")
        self.scalar_attribute_test(ts,parent,"unicode","string",u"hello world")
        self.scalar_attribute_test(ts,parent,"flag","bool",True)

    def test_array_attribute(self,ts,parent):

        shape = (10,20)
        data = numpy.ones(shape,dtype="uint8")
        self.array_attribute_test(ts,parent,"uint8_attr","uint8",shape,data)
        data = numpy.ones(shape,dtype="int8")
        self.array_attribute_test(ts,parent,"int8_attr","int8",shape,data)
        
        data = numpy.ones(shape,dtype="uint16")
        self.array_attribute_test(ts,parent,"uint16_attr","uint16",shape,data)
        data = numpy.ones(shape,dtype="int16")
        self.array_attribute_test(ts,parent,"int16_attr","int16",shape,data)
        
        data = numpy.ones(shape,dtype="uint32")
        self.array_attribute_test(ts,parent,"uint32_attr","uint32",shape,data)
        data = numpy.ones(shape,dtype="int32")
        self.array_attribute_test(ts,parent,"int32_attr","int32",shape,data)
        
        if platform.architecture()[0] != '32bit':
            data = numpy.ones(shape,dtype="uint64")
            self.array_attribute_test(ts,parent,"uint64_attr","uint64",shape,data)
            data = numpy.ones(shape,dtype="int64")
            self.array_attribute_test(ts,parent,"int64_attr","int64",shape,data)

        data = numpy.ones(shape,dtype="float32")
        self.array_attribute_test(ts,parent,"float32_attr","float32",shape,data)
        data = numpy.ones(shape,dtype="float64")
        self.array_attribute_test(ts,parent,"float64_attr","float64",shape,data)

        if platform.architecture()[0] != '32bit':
            data = numpy.ones(shape,dtype="float128")
            self.array_attribute_test(ts,parent,"float128_attr","float128",shape,data)
        
        data = numpy.ones(shape,dtype="complex64")
        self.array_attribute_test(ts,parent,"complex32_attr","complex32",shape,data)
        data = numpy.ones(shape,dtype="complex128")
        self.array_attribute_test(ts,parent,"complex64_attr","complex64",shape,data)

        if platform.architecture()[0] != '32bit':
            data = numpy.ones(shape,dtype="complex256")
            self.array_attribute_test(ts,parent,"complex128_attr","complex128",shape,data)

        data = numpy.ones(shape,dtype="bool")
        self.array_attribute_test(ts,parent,"flag_attr","bool",shape,data)
        




