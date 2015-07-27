
import pni.core
import unittest
import utils_test
import numpy

class test_utils(unittest.TestCase):
    def test_list_from_vector(self):
        l = utils_test.list_from_vector();
        self.assertEqual(len(l),4)

        index = 1
        for x in l:
            self.assertEqual(x,index)
            index += 1

    def test_list_from_list(self):
        l = utils_test.list_from_list();
        self.assertEqual(len(l),4)

        index = 1
        for x in l:
            self.assertEqual(x,index)
            index += 1

    def test_vector_from_list(self):
        l = [0,1,2,3]
        self.assertTrue(utils_test.vector_from_list(l))

    def test_is_unicode(self):
        self.assertTrue(utils_test.is_unicode(u"hello world"))

        self.assertTrue(not utils_test.is_unicode("hello world"))

    def test_unicode2str(self):

        self.assertTrue(not utils_test.is_unicode(
                        utils_test.unicode2str(u"hello world")))
    
    def test_is_int(self):
        self.assertTrue(utils_test.is_int(1))
        self.assertTrue(not utils_test.is_int(1+2j))
        self.assertTrue(not utils_test.is_int(1.2))
        self.assertTrue(not utils_test.is_int(numpy.arange(1,20)))
        self.assertTrue(not utils_test.is_int(u"1"))
        self.assertTrue(not utils_test.is_int("1"))

    def test_is_long(self):
        self.assertTrue(not utils_test.is_long(1))
        self.assertTrue(utils_test.is_long(long(1)))
        self.assertTrue(not utils_test.is_long(1.2))
        self.assertTrue(not utils_test.is_long(1+3j))
        self.assertTrue(not utils_test.is_long(u"1"))
        self.assertTrue(not utils_test.is_long("1"))
        self.assertTrue(not utils_test.is_long(numpy.arange(1,20)))

    def test_is_float(self):
        self.assertTrue(utils_test.is_float(1.2))
        self.assertTrue(not utils_test.is_float(1))
        self.assertTrue(not utils_test.is_float(1+3j))
        self.assertTrue(not utils_test.is_float(u"1.2"))
        self.assertTrue(not utils_test.is_float("1.2"))
        self.assertTrue(not utils_test.is_float(numpy.arange(1,10)))

    def test_is_bool(self):
        self.assertTrue(utils_test.is_bool(True))
        self.assertTrue(utils_test.is_bool(False))

        self.assertTrue(not utils_test.is_bool(1))
        self.assertTrue(not utils_test.is_bool(long(1)))
        self.assertTrue(not utils_test.is_bool(1.2))
        self.assertTrue(not utils_test.is_bool(1+32j))
        self.assertTrue(not utils_test.is_bool(u"1"))
        self.assertTrue(not utils_test.is_bool("1"))
        self.assertTrue(not utils_test.is_bool(numpy.arange(1,10)))

    def test_is_complex(self):
        self.assertTrue(utils_test.is_complex(1+3j))
        
        self.assertTrue(not utils_test.is_complex(1))
        self.assertTrue(not utils_test.is_complex(long(1)))
        self.assertTrue(not utils_test.is_complex(1.2))
        self.assertTrue(not utils_test.is_complex(u"1"))
        self.assertTrue(not utils_test.is_complex("1"))
        self.assertTrue(not utils_test.is_complex(numpy.arange(1,10)))

    def test_is_string(self):
        self.assertTrue(utils_test.is_string("hello world"))

        self.assertTrue(not utils_test.is_string(1))
        self.assertTrue(not utils_test.is_string(long(1)))
        self.assertTrue(not utils_test.is_string(1.2))
        self.assertTrue(not utils_test.is_string(True))
        self.assertTrue(not utils_test.is_string(u"1"))
        self.assertTrue(not utils_test.is_string(numpy.arange(1,10)))

    def test_is_scalar(self):
        self.assertTrue(utils_test.is_scalar(1))
        self.assertTrue(utils_test.is_scalar(True))
        self.assertTrue(utils_test.is_scalar(1.2))
        self.assertTrue(utils_test.is_scalar(long(2)))
        self.assertTrue(utils_test.is_scalar("hello"))
        self.assertTrue(utils_test.is_scalar(u"hello"))
        self.assertTrue(utils_test.is_scalar(1+3j))




