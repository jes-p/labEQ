"""
Point source inversion module for LabDataSet objects.
"""

import matplotlib.pyplot as plt
import obspy
# from obspy.signal import cross_correlation
import numpy as np
from scipy import signal  # for filtering
import glob
import time

# Add paths to my other modules
import sys
import os

_home = os.path.expanduser("~") + "/"
for d in os.listdir(_home + "git_repos"):
    sys.path.append(_home + "git_repos/" + d)

import lab_data_set as lds

# import GF accessor. or maybe don't, just require GF as arg and deal outside
# no, I'd like to have pointsource handle requesting the right GF from the accessor

def invert(ds: lds.LabDataSet, event_num, weights: dict):
    # check if lds object passed has instrument response, maybe warn if not

