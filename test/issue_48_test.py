#regresssion test for issue 48

import unittest
import numpy

import pni.io.nx.h5 as nx

class Issue_48_Test(unittest.TestCase):
    def setUp(self):
        self.nxfile = nx.create_file("issue_48_test.nx",overwrite=True)
        self.root   = self.nxfile["/"]

    def tearDown(self):
        self.root.close()
        self.nxfile.close()

    def run_test(self,tc,scalar):
        f = self.root.create_field("test"+tc,tc)
        
        f[0] = scalar()

    def test_issue_int(self):
        self.run_test("uint8",numpy.uint8)
        self.run_test("int8",numpy.int8)
        self.run_test("uint16",numpy.uint16)
        self.run_test("int16",numpy.int16)
        self.run_test("uint32",numpy.uint32)
        self.run_test("int32",numpy.int32)
        self.run_test("uint64",numpy.uint64)
        self.run_test("int64",numpy.int64)

    def test_issue_float(self):
        self.run_test("float32",numpy.float32)
        self.run_test("float64",numpy.float64)
        self.run_test("float128",numpy.float128)
    
    def test_issue_complex(self):
        self.run_test("complex32",numpy.complex64)
        self.run_test("complex64",numpy.complex128)
        self.run_test("complex128",numpy.complex256)

    def test_issue_bool(self):
        self.run_test("bool",numpy.bool8)



