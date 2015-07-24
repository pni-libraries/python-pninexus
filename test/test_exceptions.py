import unittest

import pni.core
import ex_trans_test;

class test_exceptions(unittest.TestCase):
    def test_throw_memory_allocation_error(self):
        self.assertRaises(pni.core.memory_allocation_error,
                          ex_trans_test.throw_memory_allocation_error)

    def test_throw_memory_not_allocated_error(self):
        self.assertRaises(pni.core.memory_not_allocated_error,
                          ex_trans_test.throw_memory_not_allocated_error)
    
    def test_throw_shape_mismatch_error(self):
        self.assertRaises(pni.core.shape_mismatch_error,
                          ex_trans_test.throw_shape_mismatch_error)
    
    def test_throw_size_mismatch_error(self):
        self.assertRaises(pni.core.size_mismatch_error,
                          ex_trans_test.throw_size_mismatch_error)

    def test_throw_index_error(self):
        self.assertRaises(pni.core.index_error,
                          ex_trans_test.throw_index_error)

    def test_throw_key_error(self):
        self.assertRaises(pni.core.key_error,
                          ex_trans_test.throw_key_error)

    def test_throw_file_error(self):
        self.assertRaises(pni.core.file_error,
                          ex_trans_test.throw_file_error)

    def test_throw_type_error(self):
        self.assertRaises(pni.core.type_error,
                          ex_trans_test.throw_type_error)

    def test_throw_value_error(self):
        self.assertRaises(pni.core.value_error,
                          ex_trans_test.throw_value_error)

    def test_throw_range_error(self):
        self.assertRaises(pni.core.range_error,
                          ex_trans_test.throw_range_error)

    def test_throw_not_implemented_error(self):
        self.assertRaises(pni.core.not_implemented_error,
                          ex_trans_test.throw_not_implemented_error)

    def test_throw_iterator_error(self):
        self.assertRaises(pni.core.iterator_error,
                          ex_trans_test.throw_iterator_error)
    
    def test_throw_cli_argument_error(self):
        self.assertRaises(pni.core.cli_argument_error,
                          ex_trans_test.throw_cli_argument_error)
    
    def test_throw_cli_error(self):
        self.assertRaises(pni.core.cli_error,
                          ex_trans_test.throw_cli_error)

