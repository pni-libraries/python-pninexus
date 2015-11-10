import unittest
import numpy
import os

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import nxfield
from pni.io.nx.h5 import deflate_filter
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io import ObjectError

#implementing test fixture
class iteration_test_on_group(unittest.TestCase):
    """
    Test the attribute iterator on the attributes created on a field
    """
    file_path = os.path.split(__file__)[0]
    file_name = "iteration_test_on_field.nxs"
    full_path = os.path.join(file_path,file_name)

    attr_names = ["attr_uint8","attr_int8","attr_uint16","attr_int16",
                  "attr_uint32","attr_int32","attr_uint64","attr_int64",
                  "attr_float32", "attr_float64","attr_float128",
                  "attr_complex32","attr_complex64","attr_complex128",
                  "attr_string","attr_bool","NX_class"]
    
    def setUp(self):
        self.gf = create_file(self.full_path,overwrite=True)

        self.group = self.gf.root().create_group("entry","NXentry") 
        self.group.attributes.create("attr_uint8","uint8")
        self.group.attributes.create("attr_int8","int8")
        self.group.attributes.create("attr_uint16","uint16")
        self.group.attributes.create("attr_int16","int16")
        self.group.attributes.create("attr_uint32","uint32")
        self.group.attributes.create("attr_int32","int32")
        self.group.attributes.create("attr_uint64","uint64")
        self.group.attributes.create("attr_int64","int64")
        self.group.attributes.create("attr_float32","float32")
        self.group.attributes.create("attr_float64","float64")
        self.group.attributes.create("attr_float128","float128")
        self.group.attributes.create("attr_complex32","complex32")
        self.group.attributes.create("attr_complex64","complex64")
        self.group.attributes.create("attr_complex128","complex128")
        self.group.attributes.create("attr_string","string")
        self.group.attributes.create("attr_bool","bool")

    def tearDown(self):
        self.group.close()
        self.gf.close()


    def test_iterator(self):
        
        self.assertEqual(self.group.attributes.size,len(self.attr_names))
        for a in self.group.attributes:
            self.attr_names.remove(a.name)

        self.assertEqual(len(self.attr_names),0)

