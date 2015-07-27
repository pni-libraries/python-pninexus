
import pni.core
import unittest
import numpy_utils_test as nu_test
import numpy

class test_numpy_utils(unittest.TestCase):
    def test_is_array(self):
        self.assertTrue(nu_test.is_array(numpy.arange(1,10)))
        self.assertTrue(not nu_test.is_array(1))
        self.assertTrue(not nu_test.is_array("hello world"))
    
    def test_is_scalar(self):
        self.assertTrue(nu_test.is_scalar(numpy.array(1)))
        self.assertTrue(not nu_test.is_scalar(numpy.array([1,2,3])))
        self.assertTrue(not nu_test.is_scalar(1))

    def test_check_type_id_from_object(self):
        self.assertTrue(nu_test.check_type_id_uint8_from_object(numpy.array(1,dtype="uint8")))
        self.assertTrue(nu_test.check_type_id_int8_from_object(numpy.array(1,dtype="int8")))
        self.assertTrue(nu_test.check_type_id_uint16_from_object(numpy.array(1,dtype="uint16")))
        self.assertTrue(nu_test.check_type_id_int16_from_object(numpy.array(1,dtype="int16")))
        self.assertTrue(nu_test.check_type_id_uint32_from_object(numpy.array(1,dtype="uint32")))
        self.assertTrue(nu_test.check_type_id_int32_from_object(numpy.array(1,dtype="int32")))
        self.assertTrue(nu_test.check_type_id_uint64_from_object(numpy.array(1,dtype="uint64")))
        self.assertTrue(nu_test.check_type_id_int64_from_object(numpy.array(1,dtype="int64")))
        self.assertTrue(nu_test.check_type_id_float32_from_object(numpy.array(1,dtype="float32")))
        self.assertTrue(nu_test.check_type_id_float64_from_object(numpy.array(1,dtype="float64")))
        self.assertTrue(nu_test.check_type_id_float128_from_object(numpy.array(1,dtype="float128")))
        self.assertTrue(nu_test.check_type_id_complex32_from_object(numpy.array(1,dtype="complex64")))
        self.assertTrue(nu_test.check_type_id_complex64_from_object(numpy.array(1,dtype="complex128")))
        self.assertTrue(nu_test.check_type_id_complex128_from_object(numpy.array(1,dtype="complex256")))
        self.assertTrue(nu_test.check_type_id_string_from_object(numpy.array("hello",dtype="string")))
        self.assertTrue(nu_test.check_type_id_bool_from_object(numpy.array(True,dtype="bool")))
    
    def test_check_type_str_from_object(self):
        self.assertTrue(nu_test.check_type_str_uint8_from_object(numpy.array(1,dtype="uint8")))
        self.assertTrue(nu_test.check_type_str_int8_from_object(numpy.array(1,dtype="int8")))
        self.assertTrue(nu_test.check_type_str_uint16_from_object(numpy.array(1,dtype="uint16")))
        self.assertTrue(nu_test.check_type_str_int16_from_object(numpy.array(1,dtype="int16")))
        self.assertTrue(nu_test.check_type_str_uint32_from_object(numpy.array(1,dtype="uint32")))
        self.assertTrue(nu_test.check_type_str_int32_from_object(numpy.array(1,dtype="int32")))
        self.assertTrue(nu_test.check_type_str_uint64_from_object(numpy.array(1,dtype="uint64")))
        self.assertTrue(nu_test.check_type_str_int64_from_object(numpy.array(1,dtype="int64")))
        self.assertTrue(nu_test.check_type_str_float32_from_object(numpy.array(1,dtype="float32")))
        self.assertTrue(nu_test.check_type_str_float64_from_object(numpy.array(1,dtype="float64")))
        self.assertTrue(nu_test.check_type_str_float128_from_object(numpy.array(1,dtype="float128")))
        self.assertTrue(nu_test.check_type_str_complex32_from_object(numpy.array(1,dtype="complex64")))
        self.assertTrue(nu_test.check_type_str_complex64_from_object(numpy.array(1,dtype="complex128")))
        self.assertTrue(nu_test.check_type_str_complex128_from_object(numpy.array(1,dtype="complex256")))
        self.assertTrue(nu_test.check_type_str_string_from_object(numpy.array("hello",dtype="string")))
        self.assertTrue(nu_test.check_type_str_bool_from_object(numpy.array(True,dtype="bool")))

    def test_get_shape(self):
        s = nu_test.get_shape(numpy.ones((2,5),dtype="uint8"))
        self.assertEqual(len(s),2)
        self.assertEqual(s[0],2)
        self.assertEqual(s[1],5)

        s = nu_test.get_shape(numpy.ones((10),dtype="int16"))
        self.assertEqual(len(s),1)
        self.assertEqual(s[0],10)
        
        s = nu_test.get_shape(numpy.array(10))
        self.assertEqual(len(s),0)

        self.assertRaises(pni.core.type_error,nu_test.get_shape,1)

    def test_get_size(self):
        self.assertEqual(nu_test.get_size(numpy.ones((2,5),dtype="uint8")),10)
        self.assertEqual(nu_test.get_size(numpy.ones((10),dtype="int16")),10)
        self.assertEqual(nu_test.get_size(numpy.array(10)),1)


