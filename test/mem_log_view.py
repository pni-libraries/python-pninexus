#!/usr/bin/env python 

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy
import sys
import argparse

parser = argparse.ArgumentParser(description="Memory consumption analyzer")
parser.add_argument("-i","--ignore",dest="ignore",
                    default=0,
                    type=int,
                    action="store",
                    help="number of data points to ignore (from the start)",
                    metavar="NIGNORE")
parser.add_argument("logfile",action="store",type=str,
                    help="Logfile to read",
                    metavar="LOGFILE")



args = parser.parse_args()

if not args.logfile:
    print("You have to pass a logfile as an argument")
    sys.exit(1)


data = numpy.loadtxt(args.logfile)
run = data[args.ignore:-1,0]
tmem = data[args.ignore:-1,1]
rmem = data[args.ignore:-1,2]

print("start memory : ",rmem[0])
print("stop  memory : ",rmem[-1])
print("bytes per run: ",(rmem[-1]-rmem[0])/rmem.size)

plt.figure()
plt.plot(run,rmem)
plt.title("{logfile}\n resident memory consumption".format(logfile=args.logfile))
plt.xlabel("runs")
plt.ylabel("memory (bytes)")
plt.show()




