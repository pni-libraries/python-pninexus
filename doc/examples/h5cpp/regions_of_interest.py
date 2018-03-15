#
# example showing how to read regions of interest (2d selection in a block)
#
from __future__ import print_function
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.property import LinkCreationList, DatasetCreationList, DatasetLayout
from pninexus.h5cpp.dataspace import Hyperslab,UNLIMITED,Simple
from matplotlib import pyplot
import numpy

class UniformRandomVariable(object):
    
    def __init__(self,x0,x_min,x_max):
        
        self.x0 = x0
        self.x_min = x_min
        self.x_max = x_max
        
    def __call__(self):
        
        offset = 

def gauss2d(x,y,x0,y0,sx,sy):
    
    return numpy.exp(-(
        (x[numpy.newaxis,:]-x0)**2/2./sx**2 - (y[:,numpy.newaxis]-y0)**2/2.0/sy**2
        ))





#
# define regions of interest 
#
roi1 = ((10,10,),(20,20))
roi2 = ((40,40,),(60,60))
rois = (roi1,roi2)

