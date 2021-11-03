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
from os.path import expanduser
home = expanduser('~')+'/'
sys.path.append(home+'dev/')

from ..utils import utils, green

# import GF accessor. or maybe don't, just require GF as arg and deal outside
# no, I'd like to have pointsource handle requesting the right GF from the accessor

def ball_forward(ds, event_id, pre=200, tot_len=2048, diam=0.75e-3, plot_title=''):
    """Produce forward models for a ball drop. Plot is plot_title is given. Return dict of synthetics.
    """
    trcs = ds.get_event_traces(event_id, pre=pre, omit=False)
    # get the GF
    dists = ds.get_dists(event_id)
    gf = green.get_greens(dists,tot_len-pre,pre,source_type='force')
    # get ball force
    _,ballforce = utils.ball_force(diam=diam)
    # make synths
    synths = {stn:np.convolve(np.diff(gf[stn]['f33']),ballforce) for stn in ds.stns}
    if plot_title:
        # grid_plot to compare
        plotdata = {}
        for stn in ds.stns:
            # zeroed traces
            zt = trcs[stn]-np.mean(trcs[stn][:pre-20])
            plotdata[stn] = [{'y':zt, 'name':'zeroed trace'}]
            # synths
            calfac = np.min(zt[:pre+len(ballforce)])/np.min(synths[stn][:pre+len(ballforce)])
            plotdata[stn].append({'y':synths[stn]*calfac, 'name':'synth'})
        utils.grid_plot(plotdata,plot_title)
    return synths

def invert(ds, event_num, weights: dict, pre=200, tot_len=2048, extra = 500, dt = 2.5e-8, filt=[1e4,1e7]):
    """Run a point source inversion on the event."""
    
    event_str = f'event_{event_num:0>3d}'
    # get tag from event
    tag,trace_num = ds.auxiliary_data.LabEvents[event_str].parameters['LabPicks'].split('/')[:2]
    trace_num = int(trace_num[2:])
    # prepare plot_data dict to collect pieces throughout the analysis
    plot_data = {}
    
    # Get and prep records
    ## pull recs extra-long to filter better (dict of {'AExx': np.array})
    traces = ds.get_traces(tag, trace_num, event_str=event_str, pre=pre+extra, tot_len=tot_len+2*extra)
    # make a shifted time array for plotting extra-long recs
    x_raw = np.array(range(tot_len+2*extra)) - extra
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
        # add raw and prepped traces to plot_data
        plot_data[stn] = []
        plot_data[stn].append({'x':x_raw, 'y':traces[stn],'name':'raw','legendgroup':'raw'})
        plot_data[stn].append({'y':prep_trcs[stn],'name':'pre-proc','legendgroup':'pre-proc'})
    # Checkpoint: plot original and prepped traces
    utils.grid_plot(plot_data,'inversion_prep')
    
    
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
