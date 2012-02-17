import sys
import unittest


from pni.utils import Array
from pni.utils import Float32Scalar



sys.path.insert(0,"../pni")

from nx.h5 import NXFile


#implementing test fixture
class NXFileTest(unittest.TestCase):
    def setUp(self):
        self.gf = NXFile()
        self.gf.overwrite = True
        self.gf.readonly = False
        self.gf.filename = "NXFileTest.h5"

        self.gf.create()

    def tearDown(self):
        self.gf.close()

    def test_creation(self):
        f = NXFile()

        f.filename = "test.h5"
        f.overwrite = True
        f.read_only = True

        f.create()
        self.assertTrue(f.is_open())
        f.close()
    
        f.overwrite = False
        self.assertRaises(UserWarning,f.create)

    def test_attributes(self):
        s = "a string attribute"
        self.gf.set_attr("text","a string attribute")
        g = self.gf.create_group("data")
        print "createed group", g

        s = Float32Scalar("sca","a.u.","a test scalar")
        s.value = 100.
        self.gf.set_attr("sca",s)

        a = Array(shape=(10,5),dtype="uint16")
        #self.gf.set_attr("det",a)

