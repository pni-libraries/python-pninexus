#
# example showing how to read regions of interest (2d selection in a block)
#
from __future__ import print_function
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.property import LinkCreationList, DatasetCreationList, DatasetLayout
from pninexus.h5cpp.dataspace import Hyperslab,UNLIMITED,Simple
from pninexus.h5cpp.node import Dataset
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection

import numpy

class UniformRandomVariable(object):
    
    def __init__(self,x0,x_min,x_max):
        
        self.x0 = x0
        self.x_min = x_min
        self.x_max = x_max
        self._current = self.__call__()
        
    @property
    def current(self):
        return self._current
        
    def __call__(self):
        
        offset = self.x_min+(self.x_max-self.x_min)*numpy.random.random_sample()
        self._current = offset+self.x0
        return self._current

def gauss2d(x,y,i0,x0,y0,sx,sy):
    
    return i0*numpy.exp(-(
        (x[numpy.newaxis,:]-x0)**2/2./sx**2 + (y[:,numpy.newaxis]-y0)**2/2.0/sy**2
        ))
    
class Noise(object):
    def __init__(self,av,min,max):
        self._av = av
        self._min = min
        self._max = max
        
    def __call__(self,x,y):
        
        return self._av + (self._min + (self._max-self._min)*numpy.random.rand(len(y),len(x)))
        
    
class Peak(object):
    def __init__(self,i0,x0,y0,sx,sy):
        
        self._i0 = i0
        self._x0 = x0
        self._y0 = y0
        self._sx = sx
        self._sy = sy
        
    def propagate(self):
        self._i0()
        self._x0()
        self._y0()
        self._sx()
        self._sy()
            
    @property
    def i0(self):
        return self._i0.current
    
    @property
    def x0(self):
        return self._x0.current
    
    @property
    def y0(self):
        return self._y0.current
    
    @property
    def sx(self):
        return self._sx.current
    
    @property
    def sy(self):
        return self._sy.current
        
    def __call__(self,x,y):
        
        return gauss2d(x,y,self.i0,self.x0,self.y0,self.sx,self.sy)

def create_dataset(base,nx,ny):
    
    datatype = h5cpp.datatype.kFloat64
    dataspace = Simple((0,ny,nx),(UNLIMITED,ny,nx))
    lcpl = LinkCreationList()
    dcpl = DatasetCreationList()
    dcpl.layout = DatasetLayout.CHUNKED
    dcpl.chunk = (1,ny,nx)
    
    return Dataset(base,h5cpp.Path("data"),datatype,dataspace,lcpl,dcpl)



xchannels = numpy.arange(0,1024)
ychannels = numpy.arange(0,512)
noise = Noise(10,-5,5)
peak1 = Peak(i0 = UniformRandomVariable(100,-50,50),
             x0 = UniformRandomVariable(300,-100,100),
             y0 = UniformRandomVariable(200,-10,10),
             sx = UniformRandomVariable(20,-5,10),
             sy = UniformRandomVariable(100,-50,50))
peak2 = Peak(i0 = UniformRandomVariable(300,-100,100),
             x0 = UniformRandomVariable(745,-60,60),
             y0 = UniformRandomVariable(300,-60,60),
             sx = UniformRandomVariable(100,-30,10),
             sy = UniformRandomVariable(100,-30,30))


h5file = h5cpp.file.create("roi.h5",AccessFlags.TRUNCATE)
root   = h5file.root()
dataset = create_dataset(root,1024,512)

#
# writing data to disk
#
selection = Hyperslab(offset=(0,0,0),block=(1,512,1024))
for index in range(0,200):
    if index % 10 == 0: print("writing frame {}".format(index))
    dataset.extent(0,1)
    selection.offset(0,index)
    data = peak1(xchannels,ychannels)+peak2(xchannels,ychannels)+noise(xchannels,ychannels)
    peak1.propagate()
    peak2.propagate()
    dataset.write(data,selection)
    
#
# define regions of interest 
#
roi1 = Hyperslab(offset=(0,85,200),block=(200,300,200))
roi2 = Hyperslab(offset=(0,137,600),block=(200,300,200)) 

data_roi1 = dataset.read(selection=roi1)
data_roi2 = dataset.read(selection=roi2)
int_roi1 = data_roi1.sum(axis=(1,2))
int_roi2 = data_roi2.sum(axis=(1,2))

    


pyplot.figure()
pyplot.title("Intensity fluctuations")
pyplot.plot(int_roi1)
pyplot.plot(int_roi2)
pyplot.legend(("ROI1","ROI2"))
    
pyplot.show()
#
# reading ROIs in a single step
#



