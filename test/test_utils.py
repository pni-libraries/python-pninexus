
import pni.core
import unittest
import utils_test

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



