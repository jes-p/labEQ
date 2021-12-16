#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lab seismology extension for the python implementation of the Adaptable Seismic Data
Format (ASDF).

TODO:
    add copyright and license?
"""

# Import base module (TODO: whole thing or just ASDFDataSet? Do I want access to errors?)
import pyasdf

# Import ObsPy to this namespace as well for methods and precision
import obspy

obspy.UTCDateTime.DEFAULT_PRECISION = 9

# Import plotting libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

# Imports for picking
import peakutils

# Import for parsing inputs (pick_all_near)
import re

# Add paths to my other modules
from .utils import utils, green
#import utils.utils as utils
#import utils.green as green
import sys
import os

_home = os.path.expanduser("~") + "/"
for d in os.listdir(_home + "dev"):
    sys.path.append(_home + "dev/" + d)
    
import warnings

######### personal helpers ##########
# Specific to my data flow, maybe separate out to a different module


def create_stream_tpc5(tpc5_path: str, ord_stns: list, net="L0", chan="XHZ", hz=40e6):
    """
    Create an obspy Stream from a .tpc5 data file.
    ord_stns must be the list of stations in the order they appear in the tpc5 file.
    """
    import h5py
    import tpc5

    def get_stats(name, net="L0", chan="XHZ", hz=40e6):
        statn_stats = obspy.core.Stats()
        statn_stats.network = net
        statn_stats.channel = chan
        statn_stats.location = "00"
        statn_stats.sampling_rate = hz
        statn_stats.station = name
        return statn_stats

    f = h5py.File(tpc5_path, "r")

    # we need to know how many blocks to read
    # all channels have the same number of blocks, use channel 1
    chan_grp = f[tpc5.getChannelGroupName(1)]
    nblocks = len(chan_grp["blocks"].keys())
    ntr = len(f["measurements"]["00000001"]["channels"])
    source_stream = obspy.Stream()

    # TODO: stop trusting the save order, implement some explicit A1:AE05 map (tpc5-sxml)
    # iterate through stations, in whatever order they were saved
    # input saved channels as chan_nums because the tpc5 has no info about which channels were saved
    # tpc5 channels will always start at 1 and increase monotonically
    for tr in range(ntr):
        statn_stats = get_stats(ord_stns[tr], net=net, chan=chan, hz=hz)

        # iterate through continuous data segments
        # TranAX calls these Blocks, obspy calls them Traces
        for blk in range(1, nblocks + 1):
            # get the trace start time
            statn_stats.starttime = (
                obspy.UTCDateTime(
                    tpc5.getStartTime(f, 1)
                )  # gives the start of the whole recording
                + tpc5.getTriggerTime(f, 1, block=blk)  # seconds from start to trigger
                - tpc5.getTriggerSample(f, 1, block=blk)
                / statn_stats.sampling_rate  # seconds from trigger to block start
            )

            # get the raw voltage data
            raw = tpc5.getVoltageData(f, tr + 1, block=blk)
            # give the stats the length, otherwise it takes 0 points
            statn_stats.npts = len(raw)
            source_stream += obspy.Trace(raw, statn_stats)
    return source_stream


def setup_experiment_from_dir(exp_name: str, glob_str="*.tpc5"):
    """Auto setup new ASDF file based on files in this directory.
    exp_name :: str name of experiment, will create exp_name.h5 file
    glob_str :: limit data files read in by glob"""
    import glob

    # initialize dataset file
    ds = LabDataSet(exp_name + ".h5", compression="gzip-3")
    # find and add stations
    statxml_fp = glob.glob("*stations.xml")
    if not len(statxml_fp):
        raise Exception("No station xml found matching pattern *stations.xml")
    elif len(statxml_fp) > 1:
        raise Warning(
            "Warning: more than one station xml found! using {}".format(statxml_fp[0])
        )
    ds.add_local_locations(statxml_fp[0])
    # stat_locs and stns properties produced automatically now
    # find and add waveforms from tpc5 files
    wf_files = glob.glob(glob_str)  # doesn't add full path prefix
    ds.all_tags = []
    print("Adding waveforms from: ")
    for wf in wf_files:
        print(wf)
        tag = wf[:-5].lower()
        ds.add_waveforms(create_stream_tpc5(wf, ds.stns), tag)
    return ds

class LabDataSet(pyasdf.ASDFDataSet):
    """
    Object handling special Lab ASDF files and operations.
    """

    def add_event(self, tag, trace_num):
        try:
            events = self.auxiliary_data.LabEvents.list()
            events.sort() # likely redundant
            next_ev = int(events[-1][-3:])+1
        except:
            # create first event
            next_ev = 0
        event_str = f'event_{next_ev:0>3d}'
        self.add_auxiliary_data(data=np.array([]),
                                data_type="LabEvents",
                                path=event_str,
                                parameters={'tag':tag, 'trace_num':trace_num})
        return event_str
    
    def add_local_locations(self, statxml_filepath):
        """
        Add stations from a StationXML inventory which must have local locations as
        'extra' info. TODO: add statxml creator and reference here
        """
        inv = obspy.read_inventory(statxml_filepath, format="stationxml")
        nsta = len(inv[0].stations)
        stat_locs = {}
        stats_order = []  # retain A1-D4 order with AExx station codes
        for sta in inv[0].stations:
            sta_code = sta.code
            stats_order.append(sta_code)
            sta_loc = (
                [
                    float(sta.extra.local_location.value[xyz].value)
                    for xyz in ["x", "y", "z"]
                ]
                if hasattr(sta, "extra")
                else (np.NaN, np.Nan, np.Nan)
            )
            stat_locs[sta_code] = sta_loc

        # add the local_locations as a dictionary to never worry about shuffling the stations and locations
        # data can't take a dictionary (and requires a shape), but parameters takes the dictionary just fine
        self.add_auxiliary_data(
            data=np.array(stats_order, dtype="S"),
            data_type="LabStationInfo",
            path="local_locations",
            parameters=stat_locs,
        )

    # add property to quickly access station locations
    @property
    def stat_locs(self) -> dict:
        return self.auxiliary_data.LabStationInfo.local_locations.parameters

    @property
    def stns(self) -> list:
        """List of stations in the order of the stationxml file."""
        return list(
            np.char.decode(self.auxiliary_data.LabStationInfo.local_locations.data[:])
        )

    ######## picking methods of object ########
    def add_picks(self, event_id, tag, trace_num, picks):
        """Add a dict of picks for a (tag,trcnum) path.
        Picks in the form {stn:[picks]}
        Returns any overwritten picks as a safety."""
        event_str = utils.parse_eid(event_id)
        # check for old_picks to overwrite
        try:
            old_picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][event_str].parameters
            del self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][event_str]
        except:
            old_picks = {}
        path = f"{tag}/tr{trace_num}/{event_str}"
        self.add_auxiliary_data(
            data=np.array([]),
            data_type="LabPicks",
            path=path,
            parameters=picks,
        )
        return old_picks

    def plot_picks(
        self, tag, trace_num, view_mid, view_len, new_picks=None, figname="picks_plot"
    ):
        """Produce an interactive plot of traces with numbered picks, and existing picks if present.
        Assumes 16 sensors.
        TODO: old_picks markers are too big
        """ 
        start,stop = [int(view_mid - view_len/2), int(view_mid + view_len/2)]

        # are there existing picks?
        # they would be stored as events so are there events?
        try:
            trc_events = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].list()
            # make a list of existing picked events
            trc_picks = [self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][estr].parameters for estr in trc_events]
            plot_op = 1
        except:
            plot_op = 0

        plot_data = {}
        for stn in self.stns:
            plot_data[stn] = []
            # plot trace
            trc = self.waveforms["L0_" + stn][tag][trace_num].data
            plot_data[stn].append({'y':trc[start : stop],'legendgroup':'traces','name':'raw trace'})
            
            # plot existing picks, if any in window
            if plot_op:
                for ev in trc_events:
                    ev_picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][ev].parameters
                    if ev_picks[stn][0] > start:
                        plot_data[stn].append({'x':np.array(ev_picks[stn]) - start,
                                               'y':trc[ev_picks[stn]],
                                               'mode':"markers",
                                               'marker':{"symbol": "x", "size": 10},
                                               'name':ev,
                                               'legendgroup':ev})

            # plot new picks, if any in window
            if new_picks:
                if new_picks[stn][0] > start:
                    plot_data[stn].append({'x':np.array(new_picks[stn]) - start,
                                           'y':trc[new_picks[stn]],
                                           'mode':"markers+text",
                                           'text':[str(np) for np in range(len(new_picks[stn]))],
                                           'textposition':"bottom center",
                                           'name':'new picks',
                                           'legendgroup':'new picks'})
                
       # plot the figure
        grid_plot(plot_data,figname,number_plots=True)
        print(f"Picks plot written to {figname}.html")

    def interactive_check_picks(
        self, tag, trace_num, event_id='', picks=None, view_mid=180000, view_len=40000, auto_pick_params={}):
        """Plot picks for all stations for one (tag,trcnum) and accept user adjustments.
        Auto-picks if no picks are provided or already stored (by noise, only appropriate for very clean signals).
        TODO: my notes imply that plotly is now interactive enough that I could replot after each input
        jump to icp
        """
        event_str = utils.parse_eid(event_id)
        stns = self.stns  # TODO: is this necessary?
        start = int(view_mid - view_len/2)

        # auto-pick if necessary
        # check for existing picks on the trace
        try:
            trc_events = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].list()
            print(f"This trace has events: {trc_events}")
            # make a list of existing picked events
            trc_picks = [self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][estr].parameters for estr in trc_events]
        except:
            trc_picks = []
        # check for new input picks
        if not picks:
            picks = {}
            if not trc_picks: # autopick only if no events already found
                print("Plotting new event with auto_pick_by_noise")
                for stn in stns:
                    trc = self.waveforms["L0_" + stn][tag][trace_num].data
                    picks[stn] = auto_pick_by_noise(trc, **auto_pick_params)
            else:
                picks = trc_picks[-1]
                event_str = trc_events[-1]
                print(f"Plotting picks for {event_str}")
        self.plot_picks(tag, trace_num, view_mid, view_len, picks)

        # ask for inputs
        print(
            "Adjustment actions available: s for select pick, r for repick near, m for"
            " manual pick"
        )
        print("Enter as [chan][action key][index], e.g. s0 to select pick 0")
        adjust = input("Adjust a channel? - to exit: ")
        # TODO: make better multipick options
        while adjust != "-":
            # get channel
            try:
                chan = int(adjust[:2])
                action = adjust[2]
                rest = adjust[3:]
            except:
                chan = int(adjust[0])
                action = adjust[1]
                rest = adjust[2:]
            # parse action
            if action == "s":
                # select one correct pick
                picks[stns[chan]] = [picks[stns[chan]][int(rest)]]
            elif action == "r":
                # pick near somewhere else
                trc = self.waveforms["L0_" + stns[chan]][tag][trace_num].data
                # check for rl,rr specification TODO: prob a better way to parse input
                rl,rr = [2000,2000]
                try:
                    num = int(rest)
                except: # letters included in remaining part
                    spec = re.split('(\D)',rest) # split on (and keep) letters
                    num = int(spec.pop(0))
                    while spec:
                        lett = spec.pop(0)
                        if lett == 'b':
                            rl = spec.pop(0)
                            rr = rl
                        elif lett == 'r':
                            rr = spec.pop(0)
                        elif lett == 'l':
                            rl = spec.pop(0)
                        else:
                            print("couldn't parse, no action")
                            continue
                picks[stns[chan]] = pick_near(
                    trc, num + start, reach_left=int(rl), reach_right=int(rr), thr=0.9
                )
            elif action == "m":
                # manually enter pick
                num = int(rest)
                if num == -1:
                    picks[stns[chan]] = [-1]
                else:
                    picks[stns[chan]] = [num + start]
            # move to next adjustment or exit
            adjust = input("Adjust a channel? - to exit: ")

        # add picks to new or existing event, catching and returning overwritten old_picks
        # only need to check events associated with this trace, which I already listed if they exist
        if not trc_picks:
            # definitely new event
            event_str = self.add_event(tag,trace_num)
        else:
            # need to compare existing event picks to new ones
            # I plotted existing events, just ask if I'm adjusting one
            update_event = input("Are you updating/reviewing an existing event? Enter event number or 'n' to create a new event.")
            if update_event == 'n':
                event_str = self.add_event(tag,trace_num)
            else:
                event_str = utils.parse_eid(int(update_event))
            
        old_picks = self.add_picks(event_str, tag, trace_num, picks)
        return old_picks
    
    def pick_all_near(
        self, tag, trace_num, near, reach_left=2000, reach_right=2000, thr=0.9
    ):
        """Run pick_near on all stations for one tag/trcnum at once. Return dict
        of picks to run through interactive_check_picks."""
        # no need to check for old picks now, add_picks will do that
        # run pick_near for each stn
        picks = {}
        for i, stn in enumerate(self.stns):
            trc = self.waveforms["L0_" + stn][tag][trace_num].data[near-reach_left:near+reach_right]
            picks[stn] = pick_near(trc, reach_left, reach_left, reach_right, thr, AIC=[])
            picks[stn] = [p+near-reach_left for p in picks[stn]]
        
        return picks

    ######## source location on object ########
    def locate_tag(self, tag, vp=0.274, bootstrap=False):
        """ deprecated and broken
        Locates all picked events within a tag. vp in cm/us
        No longer assumes one pick per trace."""
        import scipy.optimize as opt

        stns = self.stns

        # define curve_func internally so vp isn't a function input
        def curve_func_cm(X, a, b, c):
            t = np.sqrt((X[0] - a) ** 2 + (X[1] - b) ** 2 + 3.85 ** 2) / vp - c
            return t

        ntrcs = len(self.waveforms["L0_" + stns[0]][tag])
        for trace_num in range(ntrcs):
            # are there events?
            try:
                events = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].list()
                # don't think it's possible to get an empty list here
                # and the loop just won't run if somehow we do get an empty list
                #
            except:
                continue

            for event_str in events:
                picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][event_str].parameters
                # associate locations
                xys = [self.stat_locs[stn][:2] for stn in stns if picks[stn][0] > 0]
                # picks to times
                arrivals = [picks[stn][0] / 40 for stn in stns if picks[stn][0] > 0]
                # sort xys and arrivals by arrival
                xys, arrivals = list(zip(*sorted(zip(xys, arrivals), key=lambda xa: xa[1])))
                # run the nonlinear least squares
                model, cov = opt.curve_fit(
                    curve_func_cm,
                    np.array(xys).T,
                    np.array(arrivals) - arrivals[0],
                    bounds=(0, [500, 500, 50]),
                )
                o_ind = [int((arrivals[0] - model[-1]) * 40)]
                path = f"{tag}/tr{trace_num}/{event_str}"
                self.add_auxiliary_data(
                    data=model,
                    data_type="Origins",
                    path=path,
                    parameters={"o_ind": o_ind, "cov": cov},
                )
                # make sure event backlink exists
                try:
                    event = self.auxiliary_data.LabEvents[event_str].parameters
                    if event['Origins'] != path:
                        self.update_event(event_str, Origins=path)
                except:
                    self.update_event(event_str, Origins=path)

    def locate_event(self, event_id, vp=2.74):
        """Locate an event. vp in mm/us. Note that stn_locs are in cm but vp is converted.
        """
        import scipy.optimize as opt

        event_str,tag,trace_num = self.get_event(event_id)

        picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][event_str].parameters
        # associate locations
        xys = [self.stat_locs[stn][:2] for stn in self.stns if picks[stn][0] > 0]
        # picks to times
        arrivals = [picks[stn][0] / 40 for stn in self.stns if picks[stn][0] > 0]
        # sort xys and arrivals by arrival
        xys, arrivals = list(zip(*sorted(zip(xys, arrivals), key=lambda xa: xa[1])))
        # get fitting function
        def curve_func_cm(X, a, b, c):
            t = np.sqrt((X[0] - a) ** 2 + (X[1] - b) ** 2 + 3.85 ** 2) / (vp/10) - c
            return t
        # run the nonlinear least squares
        model, cov = opt.curve_fit(
            curve_func_cm,
            np.array(xys).T,
            np.array(arrivals) - arrivals[0],
            bounds=(0, [500, 500, 50]),
        )
        o_ind = [int((arrivals[0] - model[-1]) * 40)]
        path = f"{tag}/tr{trace_num}/{event_str}"
        self.add_auxiliary_data(
            data=model[:2],
            data_type="Origins",
            path=path,
            parameters={"o_ind": o_ind, "cov": cov},
        )
        

    def get_dists(self,event_id,prec=2):
        """Get epicentral station distances for an event, in mm.
           org and stat_locs in cm, dists returned in mm, rounded to two places by default
        """
        org = self.get_event_location(event_id)
        dists = {stn:np.round(np.sqrt(np.sum((self.stat_locs[stn][:2]-org)**2))*10,prec) for stn in self.stns}
        return dists
    
    def get_event_location(self,event_id):
        """Get location for event.
        """
        ev,tag,tr = self.get_event(event_id)
        org = self.auxiliary_data.Origins[tag][f'tr{tr}'][ev].data[:]
        return org

    ######## content check ########
    def check_auxdata(self):
        """Report on presence of LabPicks and Origins, return True if both present.
        Prints if any tags are missing from either but doesn't affect output.
        No check on traces within tags.
        TODO: remove repetition, extend to other checks
        """
        tags = sorted(self.waveform_tags)
        try:
            lp_tags = self.auxiliary_data.LabPicks.list()
            if len(lp_tags) < len(tags):
                print(
                    "Picks missing for tags: "
                    + str([t for t in tags if t not in lp_tags])
                )
        except:
            print("No LabPicks!")
            return False
        try:
            loc_tags = self.auxiliary_data.Origins.list()
            if len(loc_tags) < len(tags):
                print(
                    "Origins missing for tags: "
                    + str([t for t in tags if t not in loc_tags])
                )
        except:
            print("No Origins!")
            return False
        print("LabPicks and Origins present for all other tags.")
        return True

    def get_event(self, event_id):
        """Get event_str, tag, and trace_num for an event in the dataset"""
        event_str = utils.parse_eid(event_id)
        edict = self.auxiliary_data.LabEvents[event_str].parameters
        tag = edict['tag']
        tnum = edict['trace_num']
        return event_str, tag, tnum
    
    ######## get traces ########
    def get_event_traces(self, event_id, tag='', trace_num='', pre=200, tot_len=2048, omit=True, select_stns=[]):
        """Return a dict of short traces from a tag/trcnum based on picks. Omits un-picked traces."""
        if not tag:
            event_str,tag,trace_num = self.get_event(event_id)
        else:
            event_str = utils.parse_eid(event_id)
        traces = {}
        picks = self.get_event_picks(event_id)
        if not omit:
            dists = self.get_dists(event_id)
        for stn, pp in picks.items():
            if select_stns and stn not in select_stns: continue
            if "L" in stn[:1]:
                stn = stn[3:]  # deal with existing picks having dumb station names
            pp = pp[0]
            if pp == -1: 
                if omit:
                    continue
                else:
                    # use dist (mm) to get path length
                    travel = np.sqrt(dists[stn]**2 + 38.5**2)
                    event_ind = self.auxiliary_data.Origins[tag][f'tr{trace_num}'][event_str].parameters['o_ind']
                    pp = int(travel/2.74 * 40) + event_ind[0]
            sl = slice(pp - pre, pp - pre + tot_len)
            traces[stn] = self.waveforms["L0_" + stn][tag][trace_num].data[sl]
        return traces

    def get_trace_picks(self, tag, trace_num):
        """Shortcut to return the picks dictionary for all events in a trace."""
        events = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].list()
        return {ev:self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][ev].parameters for ev in events}

    def get_event_picks(self, event_id):
        """Shortcut to return the picks dictionary for an event."""
        event_str,tag,trace_num = self.get_event(event_id)
        return self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][event_str].parameters


######## picking helpers ########
# TODO move these to a picking module

def pick_near(trace, near, reach_left=2000, reach_right=2000, thr=0.9, AIC=[]):
    """Get a list of picks for one trace."""
    if len(AIC) == 0:
        AIC = get_AIC(trace, near, reach_left, reach_right)
    picks = (
        peakutils.indexes(AIC[5:-5] * (-1), thres=thr, min_dist=50) + 5
    )  # thres works better without zeros
    picks = list(picks + (near - reach_left))  # window indices -> trace indices
    # TODO: Do I need to remove duplicates and sort?
    return picks


def get_AIC(trace, near, reach_left=2000, reach_right=2000):
    """Calculate the AIC function used for picking. Accepts an Obspy trace or array-like data."""
    if hasattr(trace, "data"):
        window = trace.data[near - reach_left : near + reach_right]
    else:
        window = trace[near - reach_left : near + reach_right]
    AIC = np.zeros_like(window)
    for i in range(5, len(window) - 5):  # ends of window generate nan and -inf values
        AIC[i] = i * np.log10(np.var(window[:i])) + (len(window) - i) * np.log10(
            np.var(window[i:])
        )
    return AIC


def cut_at_rail(trace):
    """Find index where a signal rails."""
    dd = np.diff(trace)
    count = []
    for i, x in enumerate(dd):
        if x == 0:
            if len(count) == 0 or count[-1] == i - 1:  # first or next in a sequence
                count.append(i)
            else:  # new sequence
                count = [i]
            if len(count) > 9:
                break
    return count[0]


def auto_pick_by_noise(
    trace, noise_trig=10, noise_cut=50000, rl=3000, rr=3000, thresh=0.9
):
    """ Attempt to pick a trace without interaction, around the first point greater than noise_trig*noise_std."""
    cut = cut_at_rail(trace)
    trace = trace[:cut]
    # auto-aim for pick_near
    noise_std = np.std(trace[:noise_cut])
    near = np.argmax(trace > noise_std * noise_trig)  # argmax instead of argwhere()[0]
    try:
        picks = pick_near(trace, near, reach_left=rl, reach_right=rr, thr=thresh)
    except:
        picks = [-1]
    return picks

######## plotting helpers ########
# TODO move these to utils

def subplts(row, col, titles="default"):
    if titles == "default":
        titles = ("chan {}".format(i) for i in range(row * col))
    fig = make_subplots(row, col, print_grid=False, subplot_titles=tuple(titles))
    plotkey = list(
        zip(
            np.hstack([[i + 1] * col for i in range(row)]),
            [i + 1 for i in range(col)] * col,
        )
    )
    return fig, plotkey

def grid_plot(plot_data: dict, plot_file_name, legend=True, number_plots=False):
    """Make and save a 4x4 plotly plot. Takes a dict like {plot title: [items to plot]} and file name to save the figure. Items should be a list of dicts of keyword args accepted by go.Scattergl. Doesn't return the figure.
    Useful Scattergl keywords:
        y: required, no default (all others optional)
        x: defaults to indices
        name: legend entry for trace
        legendgroup: groups legend items to toggle together (e.g. 'Raw', 'Preprocess', 'New Picks', etc.)
        """
    if number_plots:
        titles = [stn+f" ({i})" for i,stn in enumerate(plot_data.keys())]
    else:
        titles = plot_data.keys()
    fig,plotkey = subplts(4,4,titles)
    color_list = ['black', 'red', 'blue', 'green', 'orange', 'magenta', 'gray']
    ncol = len(color_list)
    for i,ttl in enumerate(plot_data.keys()): 
        # iterate through traces
        for j,tdict in enumerate(plot_data[ttl]):
            # set default values
            trace_vals = {'line': {'color': color_list[j % ncol]},
                          }
            # default to drop legends after first plot for neatness
            if i > 0:
                trace_vals.update({'showlegend':False})
            # update (overwriting) with input values
            trace_vals.update(tdict)
            # add trace to plot
            fig.append_trace(go.Scattergl(**trace_vals),int(plotkey[i][0]),plotkey[i][1])
            
    # finish figure
    if not legend:
        fig["layout"].update(showlegend=False)
    fig.write_html(plot_file_name + '.html')
    print(f'Plotted {os.getcwd()}{plot_file_name}.html')
