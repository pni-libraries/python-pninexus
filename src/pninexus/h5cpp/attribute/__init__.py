from __future__ import print_function
import numpy

from pninexus.h5cpp import property
from pninexus.h5cpp._attribute import AttributeManager
from pninexus.h5cpp._attribute import Attribute

__all__ = ["property", "AttributeManager", "Attribute"]


def attribute__getitem__(self, index):

    data = self.read()

    return data[index]


def attribute_write(self, data):

    write_data = data
    if not isinstance(write_data, numpy.ndarray):
        write_data = numpy.array(write_data)
    if write_data.dtype.kind == 'U':
        try:
            write_data = write_data.astype("S")
        except Exception:
            if isinstance(data, numpy.ndarray) and data.shape:
                shape = data.shape
                if len(shape) > 1:
                    data = data.flatten()
                write_data = numpy.array(
                    [bytes(str(dt).encode('utf-8')) for dt in data])
                if len(shape) > 1:
                    write_data = write_data.reshape(shape)
            else:
                write_data = numpy.array(str(data).encode('utf-8'))
    elif write_data.dtype == 'bool':
        write_data = write_data.astype("int8")

    #  print("DATA", data, write_data)
    try:
        self._write(write_data)
    except RuntimeError as e:
        print(str(e))
        print(write_data, write_data.dtype)


def attribute_read(self):

    read_data = self._read()

    if isinstance(read_data, numpy.ndarray) and read_data.dtype.kind == 'S':
        return read_data.astype('U')
    else:
        return read_data


Attribute.__getitem__ = attribute__getitem__
Attribute.write = attribute_write
Attribute.read = attribute_read
