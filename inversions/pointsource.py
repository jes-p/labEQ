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
for d in os.listdir(_home + "dev"):
    sys.path.append(_home + "dev/" + d)

import lab_data_set as lds

# import GF accessor. or maybe don't, just require GF as arg and deal outside
# no, I'd like to have pointsource handle requesting the right GF from the accessor

def invert(ds: lds.LabDataSet, event_num, weights: dict, pre=200, tot_len=2048, extra = 500, dt = 2.5e-8, filt=[1e4,1e7]):
    """Run a point source inversion on the event."""
    
    event_str = f'event_{event_num:0>3d}'
    # get tag from event
    tag,trace_num = ds.auxiliary_data.LabEvents[event_str].parameters['LabPicks'].split('/')[:2]
    trace_num = int(trace_num[2:])
    
    # Get and prep records
    ## pull recs extra-long to filter better (dict of {'AExx': np.array})
    traces = ds.get_traces(tag, trace_num, event_str=event_str, pre=pre+extra, tot_len=tot_len+2*extra)
    ## de-mean, filter, and slice
    prep_trcs = traces.copy() # so that I can return the exact starting traces
    nyquist = 1/(2*dt)
    wl,wh = np.array(filt)/nyquist
    n = 2
    sos = signal.iirfilter(n, [wl,wh], output='sos')
    for stn,trc in prep_trcs.items():
        offset = np.mean(trc[:(pre+extra-10)])
        trc -= offset
        trc = signal.detrend(signal.sosfiltfilt(sos,trc)) # symmetric so arrival shouldn't shift
        prep_trcs[stn] = trc[extra:(tot_len+extra)]
    # Checkpoint: plot original and prepped traces
    
    return traces, prep_trcs
    
    # Get and prep GF
    ## get station dists
    ## get azimuths
    ## pull GF
    ## add source
    ## add instrument response
    ## add weights
    ## de-mean, filter, and slice
    
    # Window and stitch records and GF
    
    # Invert and synthesize
    
    # Check fits
    
    # Iterate
