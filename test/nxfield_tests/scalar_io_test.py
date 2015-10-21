import unittest
import numpy
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.core import SizeMismatchError
from .. data_generator import random_generator_factory
from .. import io_test_utils as iotu


#implementing test fixture
class scalar_io_test_uint8(unittest.TestCase):
    """
    Testing IO operations on a field. The possible operations are 
    
    """
    _typecode = "uint8"
    file_path = os.path.split(__file__)[0]

    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(self):
        self.file_name = "scalar_io_test_{tc}.nxs".format(tc=self._typecode)
        self.full_path = os.path.join(self.file_path,self.file_name)
        self.gf = create_file(self.full_path,overwrite=True)
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.generator = random_generator_factory(self._typecode)
        self.input_data = next(self.generator())
        self.input_array_data = numpy.array([next(self.generator())])
        self.scalar_type = iotu.scalars[self._typecode]

    #-------------------------------------------------------------------------
    def tearDown(self):
        self.root.close()
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def test_scalar_to_scalar_write_read_method(self):
        """
        Write a scalar to a field and read it back using the write() and 
        read() methods provided by the nxfield class
        """

        f = self.root.create_field("scalar_read_write",self._typecode)

        #reading and writing data from a single native python object
        f.write(self.input_data)
        output_data = f.read()

        self.assertAlmostEqual(self.scalar_type(output_data),
                               self.scalar_type(self.input_data))

    #-------------------------------------------------------------------------
    def test_scalar_array_to_scalar_write_read_method(self):
        """
        Write a numpy array of size 1 to a scalar field using the write()
        and read() methods.
        """
        
        f = self.root.create_field("scalar_array_read_write",self._typecode)

        #write data from a numpy array with a single element 
        f.write(self.input_array_data)
        output_data = f.read()
        self.assertAlmostEqual(self.scalar_type(self.input_array_data[0]),
                               self.scalar_type(output_data))

        self.assertRaises(SizeMismatchError,f.write,
                          numpy.ones((10)))

    
    #-------------------------------------------------------------------------
    def test_scalar_to_scalar_setitem_getitem(self):
        """
        Write a scalar value to a scalar field using the __getitem__ and 
        __setitem__ methods. 
        """

        f = self.root.create_field("scalar_setitem_getitem",self._typecode)

        f[...] = self.input_data
        output_data = f[...]

        self.assertAlmostEqual(self.scalar_type(output_data),
                               self.scalar_type(self.input_data))

   
    #-------------------------------------------------------------------------
    def test_scalar_array_to_scalar_setitem_getitem(self):
        """
        Write a scalar array value to a scalar field using the __getitem__ 
        and __setitem__ methods. 
        """

        f = self.root.create_field("scalar_array_setitem_getitem",self._typecode)

        f[...] = self.input_array_data
        output_data = f[...]

        self.assertAlmostEqual(self.scalar_type(output_data),
                               self.scalar_type(self.input_array_data[0]))
        

    #-------------------------------------------------------------------------
    def test_scalar_to_1Dfield_partial_individual(self):
        """
        Write and read individual data to a 1D field using partial IO. 
        """

        f = self.root.create_field("scalar_to_1D_partial_individual",
                                   self._typecode,
                                   shape=(10,))

        for (index,input_data) in zip(range(f.size),self.generator()):
            f[index] = input_data
            output_data = f[index]

            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(input_data))

    #-------------------------------------------------------------------------
    def test_scalar_array_to_1Dfield_partial_individual(self):
        """
        Write and read individual data to a 1D field using partial IO. 
        """

        f = self.root.create_field("scalar_array_to_1D_partial_individual",
                                   self._typecode,
                                   shape=(10,))

        for (index,input_data) in zip(range(f.size),self.generator()):
            f[index] = numpy.array([input_data])
            output_data = f[index]

            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(input_data))

    #-------------------------------------------------------------------------
    def test_scalar_to_1Dfield_broadcast(self):
        """
        Broadcast a single data value to a whoel field
        """

        f = self.root.create_field("scalar_to_1D_broadcast",
                                   self._typecode,shape=(10,))

        f[...] = self.input_data

        for output_data in f[...]:
            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(self.input_data))

    #-------------------------------------------------------------------------
    def test_scalar_array_to_1Dfield_broadcast(self):
        """
        Broadcast a single data value to a whoel field
        """

        f = self.root.create_field("scalar_array_to_1D_broadcast",
                                   self._typecode,shape=(10,))

        f[...] = self.input_array_data[...]

        for output_data in f[...]:
            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(self.input_array_data[0]))

    #-------------------------------------------------------------------------
    def test_scalar_to_2Dfield_partial_individual(self):
        """
        Writing individual scalars using partial IO
        """

        f = self.root.create_field("scalar_to_2d_indivdual",
                                   self._typecode,
                                   shape=(3,4))

        for i in range(3):
            for j in range(4):
                input_data = next(self.generator())
                f[i,j] = input_data
                output_data = f[i,j]
                self.assertAlmostEqual(self.scalar_type(output_data),
                                       self.scalar_type(input_data))

    #-------------------------------------------------------------------------
    def test_scalar_from_2Dfield_partial_strips(self):
        """
        Broadcast scalar data to strips and read them back
        """
        
        f = self.root.create_field("scalar_to_2d_strips",
                                   self._typecode,
                                   shape=(3,4))

        for index in range(f.shape[0]):
            input_data = next(self.generator())
            f[index,...] = input_data

            for output_data in f[index,...].flat:
                self.assertAlmostEqual(self.scalar_type(output_data),
                                       self.scalar_type(input_data))


    #-------------------------------------------------------------------------
    def test_scalar_from_2Dfield_broadcast(self):
        """
        Broadcast an individual value to a 2D field
        """
        
        f = self.root.create_field("scalar_to_2d_broadcast",
                                   self._typecode,
                                   shape=(3,4))
        f[...] = self.input_data
        
        for output_data in f[...].flat:
            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(self.input_data))



#=============================================================================
class scalar_io_test_uint16(scalar_io_test_uint8):
    _typecode = "uint16"

#=============================================================================
class scalar_io_test_uint32(scalar_io_test_uint8):
    _typecode = "uint32"

#=============================================================================
class scalar_io_test_uint64(scalar_io_test_uint8):
    _typecode = "uint64"

#=============================================================================
class scalar_io_test_int8(scalar_io_test_uint8):
    _typecode = "int8"

#=============================================================================
class scalar_io_test_int16(scalar_io_test_uint8):
    _typecode = "int16"

#=============================================================================
class scalar_io_test_int32(scalar_io_test_uint8):
    _typecode = "int32"

#=============================================================================
class scalar_io_test_int64(scalar_io_test_uint8):
    _typecode = "int64"

#=============================================================================
class scalar_io_test_float32(scalar_io_test_uint8):
    _typecode = "float32"

#=============================================================================
class scalar_io_test_float64(scalar_io_test_uint8):
    _typecode = "float64"

#=============================================================================
class scalar_io_test_float128(scalar_io_test_uint8):
    _typecode = "float128"

#=============================================================================
class scalar_io_test_complex32(scalar_io_test_uint8):
    _typecode = "complex32"

#=============================================================================
class scalar_io_test_complex64(scalar_io_test_uint8):
    _typecode = "complex64"

#=============================================================================
class scalar_io_test_complex128(scalar_io_test_uint8):
    _typecode = "complex128"

#=============================================================================
class scalar_io_test_bool(scalar_io_test_uint8):
    _typecode = "bool"

#=============================================================================
class scalar_io_test_string(scalar_io_test_uint8):
    _typecode = "string"
   
    #-------------------------------------------------------------------------
    def test_scalar_array_to_scalar_write_read_method(self):
        """
        Write a numpy array of size 1 to a scalar field using the write()
        and read() methods.
        """
        
        f = self.root.create_field("scalar_array_read_write",self._typecode)

        #write data from a numpy array with a single element 
        f.write(self.input_array_data)
        output_data = f.read()
        self.assertAlmostEqual(self.scalar_type(self.input_array_data[0]),
                               self.scalar_type(output_data))

        self.assertRaises(SizeMismatchError,f.write,
                          numpy.array(["hello","world"]))

        self.assertRaises(TypeError,f.write,1)

