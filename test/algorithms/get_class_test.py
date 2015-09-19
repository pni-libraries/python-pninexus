import unittest
import os
import numpy

from pni.io import ObjectError
from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import create_files
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import get_class


#implementing test fixture
class get_class_test(unittest.TestCase):
    filename = "get_class_test.nxs"

    def setUp(self):
        self._file = create_file(self.filename,overwrite=True)

    def tearDown(self):
        self._file.flush()
        self._file.close()


    def test_with_attribute(self):
        #this should work as the file does not exist yet
        root = self._file.root()
        a = root.attributes.create("test_scalar","float32")
        self.assertRaises(TypeError,get_class,a)

    def test_with_field(self):
        root = self._file.root()

        f = root.create_field("test_scalar","float64")
        self.assertRaises(TypeError,get_class,f)

    def test_with_group(self):
        root = self._file.root()
        g = root.create_group("entry","NXentry")

        self.assertEqual(get_class(g),"NXentry")

            










