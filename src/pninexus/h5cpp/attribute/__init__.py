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
        write_data = write_data.astype("S")
    elif write_data.dtype == 'bool':
        write_data = write_data.astype("int8")

    try:
        self._write(write_data)
    except RuntimeError:
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
