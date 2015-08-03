#!/usr/bin/env python 

from matplotlib import pyplot as plt
import numpy
import sys


try:
    lfile = sys.argv[1]
except:
    #read from stdin
    lfile = sys.stdin.readline()


data = numpy.loadtxt(lfile)
print data.shape
run = data[:,0]
tmem = data[:,1]
rmem = data[:,2]

print "start memory: ",rmem[20]
print "stop  memory: ",rmem[-1]

plt.figure()
plt.plot(run,rmem)
plt.title(lfile+"\n resident memory consumption")
plt.xlabel("runs")
plt.ylabel("memory (bytes)")
plt.show()




