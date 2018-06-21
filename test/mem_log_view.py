#!/usr/bin/env python

from __future__ import print_function
from matplotlib import pyplot as plt
# import numpy
import sys
import argparse
import pandas


# ----------------------------------------------------------------------------
# setting up and parse command line arguments
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Memory consumption analyzer")
parser.add_argument("logfile", action="store", type=str,
                    help="Logfile to read",
                    metavar="LOGFILE")
parser.add_argument("--start", dest="start_index", action="store", type=int,
                    help="Start index for doing the analysis",
                    default=0,
                    metavar="STARTIDX",
                    )
parser.add_argument("--stop", dest="stop_index", action="store", type=int,
                    help="Stop index for analysis",
                    default=-1,
                    metavar="STOPIDX"
                    )

args = parser.parse_args()

# if no logfile has been given by the user - abort the program
if not args.logfile:
    print("You have to pass a logfile as an argument")
    sys.exit(1)

# ----------------------------------------------------------------------------
# load data from the file
# ----------------------------------------------------------------------------
col_names = ["run", "total size", "resident", "share", "text", "lib", "data",
             "dirty pages"]
df = pandas.read_table(
    args.logfile, names=col_names, sep=" ", index_col="run")
df = df[args.start_index:args.stop_index]
print(df.describe())
df.pop("text")
df.pop("dirty pages")
df.pop("lib")
df.plot(subplots=True)
plt.show()
