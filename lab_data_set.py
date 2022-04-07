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
from scipy import signal

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

    def add_event(self, tag='', trace_num='', info={}, subpaths=True):
        """Add event to dataset under LabEvents aux data.
        Switching from using the event directly (still the case in calibration sets) to using subpaths of it (new default).
        If using subpaths, include info parameters dict (tag,trace,etc).
        """
        try:
            events = self.auxiliary_data.LabEvents.list()
            events.sort() # likely redundant
            next_ev = int(events[-1][-3:])+1
        except:
            # create first event
            next_ev = 0
        event_str = f'event_{next_ev:0>3d}'
        if not subpaths: # use old convention
            self.add_auxiliary_data(data=np.array([]),
                                    data_type="LabEvents",
                                    path=event_str,
                                    parameters={'tag':tag, 'trace_num':trace_num})
        else:
            self.add_auxiliary_data(data=np.array([]),
                                    data_type="LabEvents",
                                    path=f'{event_str}/info',
                                    parameters=info)
        print(f'Added {event_str}')
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
    def add_picks(self, event_id, picks, tag=None, trace_num=None):
        """Add a dict of picks for a (tag,trcnum) path.
        Picks in the form {stn:[picks]}
        Returns any overwritten picks as a safety."""
        if not tag:
            event_str,tag,trace_num = self.get_event(event_id)
        else:
            event_str = utils.parse_eid(event_id)
        # check for old_picks to overwrite
        if 'LabPicks' in self.auxiliary_data.list():
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
        else: # new subpath organization
            if 'picks' in self.auxiliary_data.LabEvents[event_str].list():
                old_picks = self.auxiliary_data.LabEvents[event_str].picks.parameters
                del self.auxiliary_data.LabEvents[event_str].picks
            self.add_auxiliary_data(
                data=np.array([]),
                data_type="LabEvents",
                path=event_str+'/picks',
                parameters=picks,
            )
        return old_picks

    def add_picks_subpath(self, event_id, picks):
        """Add dict of picks under a LabEvents/event path.
        Assumes new event and will not overwrite.
        """
        event_str = utils.parse_eid(event_id)
        self.add_auxiliary_data(data=np.array([]),
                                data_type='LabEvents',
                                path=f'{event_str}/picks',
                                parameters=picks)
        
    def add_location_subpath(self, event_id, location, origin_index, cov_res):
        """Add location information under a LabEvents/event path.
        Assumes new event and will not overwrite.
        """
        event_str = utils.parse_eid(event_id)
        self.add_auxiliary_data(data=np.array([]),
                                data_type='LabEvents',
                                path=f'{event_str}/location',
                                parameters={'loc':location, 'o_ind':origin_index, 'cov':cov_res})
        
    def plot_picks_LabPicks(
        self, tag, trace_num, view_mid, view_len, new_picks=None, figname="picks_plot", band=[1e3/20e6, 4e6/20e6]
    ):
        """Produce an interactive plot of traces with numbered picks, and existing picks if present.
        Assumes 16 sensors.
        TODO: old_picks markers are too big
        """ 
        start,stop = [int(view_mid - view_len/2), int(view_mid + view_len/2)]
        sos = signal.iirfilter(2, band, output='sos')

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
            trc = signal.sosfiltfilt(sos,trc)
            plot_data[stn].append({'y':trc[start : stop],'legendgroup':'traces','name':'raw trace'})
            
            # plot existing picks, if any in window
            if plot_op:
                for ev in trc_events:
                    ev_picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][ev].parameters
                    if len(ev_picks[stn]) > 0 and ev_picks[stn][0] > start:
                        plot_data[stn].append({'x':np.array(ev_picks[stn]) - start,
                                               'y':trc[ev_picks[stn]],
                                               'mode':"markers",
                                               'marker':{"symbol": "x", "size": 10},
                                               'name':ev,
                                               'legendgroup':ev})

            # plot new picks, if any in window
            if new_picks:
                if len(new_picks[stn]) > 0 and new_picks[stn][0] and new_picks[stn][0] > start:
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

        
    def plot_picks(
        self, tag, trace_num, view_mid, view_len, new_picks=None, figname="picks_plot", band=[1e3/20e6, 4e6/20e6]
    ):
        """Produce an interactive plot of traces with numbered picks, and existing picks if present.
        Assumes 16 sensors.
        Updated to 
        """ 
        start,stop = [int(view_mid - view_len/2), int(view_mid + view_len/2)]
        sos = signal.iirfilter(2, band, output='sos')

        # are there existing events on this trace?
        plot_op = 0 # default
        if tag in self.auxiliary_data.EventIndex.list():
            tn = f'tr{trace_num}'
            if tn in self.auxiliary_data.EventIndex[tag].parameters.keys():
                trc_events = self.auxiliary_data.EventIndex[tag].parameters[tn]
                plot_op = 1 # need to check for picks to plot from trc_events

        plot_data = {}
        for stn in self.stns:
            plot_data[stn] = []
            # plot trace
            trc = self.waveforms["L0_" + stn][tag][trace_num].data
            trc = signal.sosfilt(sos,trc)
            plot_data[stn].append({'y':trc[start : stop],'legendgroup':'traces','name':'raw trace'})
            
            # plot existing picks, if any in window
            if plot_op:
                for ev in trc_events:
                    if 'picks' in self.auxiliary_data.LabEvents[ev].list():
                        try:
                            pk = self.auxiliary_data.LabEvents[ev].picks.parameters[stn]
                        except KeyError:
                            pk = []
                        # parse many current forms of pick storage
                        if not hasattr(pk,'__len__'):
                            pk = [pk]
                        # now pk is a list but might still be empty/-1/out of range
                        if len(pk) > 0 and pk[0] > start and pk[-1] < stop:
                            plot_data[stn].append({'x':np.array(pk) - start,
                                                   'y':trc[pk],
                                                   'mode':"markers",
                                                   'marker':{"symbol": "x", "size": 8},
                                                   'name':ev,
                                                   'legendgroup':ev})

            # plot new picks, if any in window
            if new_picks:
                try:
                    pk = new_picks[stn]
                except KeyError:
                    pk = []
                if not hasattr(pk,'__len__'):
                    pk = [pk]
                if len(pk) > 0 and pk[0] > start:
                    plot_data[stn].append({'x':np.array(pk) - start,
                                           'y':trc[pk],
                                           'mode':"markers+text",
                                           'text':[str(np) for np in range(len(pk))],
                                           'textposition':"bottom center",
                                           'name':'new picks',
                                           'legendgroup':'new picks'})
                
       # plot the figure
        grid_plot(plot_data,figname,number_plots=True)
        print(f"Picks plot written to {figname}.html")
        
    def plot_event(self,event_id, figname="picks_plot", band=[1e3/20e6, 4e6/20e6]):
        """Plot picks for an event.
        """
        estr,tag,trc_num = self.get_event(event_id)
        # get aim and view_len to call plot_picks
        picks = self.get_event_picks(estr)
        pklist = []
        for pk in picks.values():
            if hasattr(pk,'__len__'):
                pklist.extend(pk)
            else:
                pklist.append(pk)
        # set aim in the middle
        half_span = (max(pklist)-min(pklist))//2
        view_mid = half_span + min(pklist)
        self.plot_picks(tag, trc_num, view_mid, half_span*2+2000, figname=figname, band=band)
        
    def interactive_check_picks(
        self, tag, trace_num, event_id='', picks=None, view_mid=180000, view_len=40000, band=[1e3/20e6, 4e6/20e6], auto_pick=False, auto_pick_params={}):
        """Plot picks for all stations for one (tag,trcnum) and accept user adjustments.
        Auto-picks if no picks are provided or already stored (by noise, only appropriate for very clean signals).
        TODO: my notes imply that plotly is now interactive enough that I could replot after each input
        jump to icp
        """
        event_str = utils.parse_eid(event_id)
        stns = self.stns  # TODO: is this necessary?
        start = int(view_mid - view_len/2)
        
        # make a filter to remove highest freq. noise and lowest freq. ringing
        sos = signal.iirfilter(2, band, output='sos')

        # 3/23/22 plot_picks will already plot existing events and autopicking was done separately
        # removing check for existing events and making autopicking default no
        
        # check for new input picks
        if not picks:
            picks = {}
            if auto_pick: # autopick only if no events already found
                print("Plotting new event with auto_pick_by_noise")
                for stn in stns:
                    trc = self.waveforms["L0_" + stn][tag][trace_num].data
                    trc = signal.sosfilt(sos,trc)
                    picks[stn] = auto_pick_by_noise(trc, **auto_pick_params)
        self.plot_picks(tag, trace_num, view_mid, view_len, picks, band=band)

        # ask for inputs
        print(
            "Adjustment actions available: s for select pick, r for repick near, m for"
            " manual pick, d for drop pick(s)"
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
                rl,rr = [500,500]
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
            elif action == "d":
                # set station/event to no picks with -1
                # TODO empty list would be better but then I'd have a bunch of logic to update (forced myself to start making both work for now)
                picks[stns[chan]] = [-1]
            # move to next adjustment or exit
            adjust = input("Adjust a channel? - to exit: ")

        # add picks to new or existing event, catching and returning overwritten old_picks
        # need to compare existing event picks to new ones
        # I plotted existing events, just ask if I'm adjusting one
        update_event = input("Are you updating/reviewing an existing event? Enter event number or 'n' to create a new event.")
        if update_event == 'n':
            event_str = self.add_event(tag,trace_num)
        else:
            event_str = utils.parse_eid(int(update_event))
            
        old_picks = self.add_picks(event_str, picks, tag, trace_num)
        return old_picks
    
    def interactive_repick_event(self, event_id, reach=200, band=[1e3/20e6, 2e6/20e6], buffer=1000):
        """Repick very locally with AIC and show with interactive_check_picks.
        """
        event_str,tag,trc_num = self.get_event(event_id)
        # make sure there are picks to view
        if 'picks' not in self.auxiliary_data.LabEvents[event_str].list():
            warnings.warn('Event has no picks.')
            return
        # get view range and repicks
        picks = self.auxiliary_data.LabEvents[event_str].picks.parameters
        pklist = []
        sos = signal.iirfilter(4,band,output='sos')
        for s,pk in picks.items():
            if not pk.any(): continue
            if hasattr(pk,'__len__'): pk = pk[0]
            if pk == -1: continue
            pklist.append(pk)
            # repick
            tr = self.waveforms['L0_'+s][tag][trc_num].data[pk-reach:pk+reach]
            tr = tr-np.mean(tr[:reach//2])
            if tr[reach] < 0: tr = tr * -1 # reverse polarity, assume pick on rise
            tr = signal.sosfilt(sos,tr)
            aic = AIC_func(tr)
            picks[s] = np.argmin(aic[5:reach]) + 5 + pk - reach
        pklist.sort()
        half_span = (pklist[-1] - pklist[0])//2
        mid = pklist[0] + half_span
        old_picks = self.interactive_check_picks(tag, trc_num, picks=picks, view_mid=mid, view_len=(half_span+buffer)*2, band=band)
        return old_picks
    
    def drop_picks(self, event_id, chan_nums):
        """Adjust picks to -1 for a list of channels. Useful for smaller events with many unpickable channels.
        """
        event_str,tag,trace_num = self.get_event(event_id)
        drops = [self.stns[i] for i in chan_nums]
        print(f'Preparing to drop picks for event {event_str} from stations {drops}')
        conf = input('y to confirm')
        if conf=='y':
            # drop picks
            picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][event_str].parameters
            for stn in drops:
                picks.update({stn:[-1]})
            old_picks = self.add_picks(event_str, tag, trace_num, picks)
        else:
            print('aborted')
        return old_picks
    
    def pick_all_near(
        self, tag, trace_num, near, reach_left=2000, reach_right=2000, thr=0.9, band=[1e3/20e6, 4e6/20e6]
    ):
        """Run pick_near on all stations for one tag/trcnum at once. Return dict
        of picks to run through interactive_check_picks."""
        # no need to check for old picks now, add_picks will do that
        # run pick_near for each stn
        picks = {}
        sos = signal.iirfilter(2, band, output='sos')
        for i, stn in enumerate(self.stns):
            trc = self.waveforms["L0_" + stn][tag][trace_num].data[near-reach_left:near+reach_right]
            trc = signal.sosfiltfilt(sos,trc)
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

    def locate_event(self, event_id, vp=2.74, save=True, picks=None):
        """Locate an event. vp in mm/us. Note that stn_locs are in cm but vp is converted.
        Returns model, origin index (rel. to start of trace), cov matrix.
        Defaults to saving the location for the event but won't overwrite.
        Option to provide a limited picks dict instead of using what's saved.
        """
        import scipy.optimize as opt

        event_str,tag,trace_num = self.get_event(event_id)

        if not picks:
            if 'LabPicks' in self.auxiliary_data.list():
                picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][event_str].parameters
            else:
                picks = self.auxiliary_data.LabEvents[event_str].picks.parameters
                # TODO: I keep writing this pick type parser over and over, wish it was a function
        # process picks
        use_picks = {}
        for s,pk in picks.items():
            if hasattr(pk,'__len__') and len(pk) > 0:
                pk = pk[0]
            if pk:
                use_picks[s] = pk
        # associate locations
        xys = [self.stat_locs[stn][:2] for stn in use_picks.keys()]
        # picks to times
        arrivals = [use_picks[stn] / 40 for stn in use_picks.keys()]
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
        if save:
            path = f"{tag}/tr{trace_num}/{event_str}"
            self.add_auxiliary_data(
                data=model[:2],
                data_type="Origins",
                path=path,
                parameters={"o_ind": o_ind, "cov": cov},
            )
        return model,o_ind,cov
    
    def locate_picks(self, picks, vp=2.74):
        """Locate an event based on given picks dict. vp in mm/us. Note that stn_locs are in cm but vp is converted.
        Now handles picks dict with direct values or lists.
        Returns model, origin index (rel. to start of trace), cov matrix.
        """
        import scipy.optimize as opt

        # process picks
        xys,arrivals = [[],[]]
        for stn,pk in picks.items():
            if hasattr(pk,'__len__'):
                if len(pk) > 0 and pk[0] and pk[0] > 0: # [], [[]], and [-1] all in use as non-picks
                    xys.append(self.stat_locs[stn][:2])
                    arrivals.append(pk[0]/40)
            else:
                if pk > 0:
                    xys.append(self.stat_locs[stn][:2])
                    arrivals.append(pk/40)
                    
        # # associate locations
        # xys = [self.stat_locs[stn][:2] for stn in picks.keys()
        #        if len(picks[stn]) > 0 and picks[stn][0] > 0]
        # # picks to times
        # arrivals = [picks[stn][0] / 40 for stn in picks.keys()
        #             if len(picks[stn]) > 0 and picks[stn][0] > 0]
        
        # sort xys and arrivals by arrival
        xys, arrivals = list(zip(*sorted(zip(xys, arrivals), key=lambda xa: xa[1])))
        # get fitting function
        def curve_func_cm(X, a, b, c):
            t = np.sqrt((X[0] - a) ** 2 + (X[1] - b) ** 2 + 3.85 ** 2) / (vp/10) - c
            return t
        # run the nonlinear least squares
        try:
            model, cov = opt.curve_fit(
                curve_func_cm,
                np.array(xys).T,
                np.array(arrivals) - arrivals[0],
                bounds=(0, [500, 500, 50]),
            )
        except RuntimeError:
            return [],[],[[100]] # stand-in outputs to fail a cov check
        
        o_ind = [int((arrivals[0] - model[-1]) * 40)]
        return model,o_ind,cov
        
    def jk_loc(self, event_id, vp=2.74):
        """Run location for event with sub-sample of picks. Return arrival times for least error or user-chosen model.
        """
        all_picks = self.get_event_picks(event_id)
        # remove one pick from each location
        locs = []
        for i,stn in enumerate(all_picks.keys()):
            picks = {s:p for s,p in all_picks.items() if s != stn}
            locs.append((i,self.locate_event(event_id,save=False,picks=picks)))
        # print table of results
        print('n\t x\t y\t xerr\t terr\t')
        for n,l in locs:
            print(f'{n}\t {l[0][0]:.1f}\t {l[0][1]:.1f}\t {l[2][0,0]:.1f}\t {l[2][-1,-1]:.1f}\t ')
        # ask which result to return
        req = input('Select result')
        if req == 't':
            # return lowest time error
            li = sorted(locs,key=lambda l:l[1][2][-1,-1])[0][0] 
        elif req == 'x':
            # return lowest x error
            li = sorted(locs,key=lambda l:l[1][2][0,0])[0][0] 
        else:
            # assume integer selection was made
            li = int(req)
            
        # return calculated arrivals for this origin as picks dict
        return self.calc_arrivals(locs[li][1][0],locs[li][1][1][0])

    def calc_arrivals(self,org,o_ind):
        """Get a picks dict of modeled arrival indices for all stations based on given location.
        """
        picks = {}
        for stn,sloc in self.stat_locs.items():
            dx,dy = [10*(sloc[i]-org[i]) for i in range(2)]
            travel = np.sqrt(dx**2 + dy**2 + 38.5**2)
            picks[stn] = [int(travel/2.74 * 40) + o_ind]
        return picks
        
    def get_dists(self,event_id,prec=2,sort=True,org=None):
        """Get epicentral station distances for an event, in mm.
           org and stat_locs in cm, dists returned in mm, rounded to two places by default
        """
        if not org.any():
            org = self.get_event_location(event_id)
        dists = {stn:np.round(np.sqrt(np.sum((self.stat_locs[stn][:2]-org)**2))*10,prec) for stn in self.stns}
        if sort:
            dists = dict(sorted(dists.items(), key=lambda sd: sd[1]))
        return dists
    
    def get_event_location(self,event_id):
        """Get location for event.
        """
        ev,tag,tr = self.get_event(event_id)
        if 'Origins' in self.auxiliary_data.list():
            org = self.auxiliary_data.Origins[tag][f'tr{tr}'][ev].data[:]
        else:
            org = self.auxiliary_data.LabEvents[ev]['location'].parameters['loc'][:2]
        return org

    def get_event_origin_index(self,event_id):
        """Get location for event.
        """
        ev,tag,tr = self.get_event(event_id)
        if 'Origins' in self.auxiliary_data.list():
            o_ind = self.auxiliary_data.Origins[tag][f'tr{tr}'][ev].parameters['o_ind']
        else:
            o_ind = self.auxiliary_data.LabEvents[ev]['location'].parameters['o_ind']
        if hasattr(o_ind,"__len__"): o_ind = o_ind[0]
        return o_ind

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

    def get_event(self, event_id, near=False):
        """Get event_str, tag, and trace_num for an event in the dataset"""
        event_str = utils.parse_eid(event_id)
        edict = self.auxiliary_data.LabEvents[event_str]['info'].parameters
        tag = edict['tag']
        tnum = edict['trace']
        if near:
            arg_out = (event_str, tag, tnum, edict['near'])
        else:
            arg_out = (event_str, tag, tnum)
        return arg_out
    
    ######## get traces ########
    def get_event_traces(self, event_id, tag='', trace_num='', pre=200, tot_len=2048, omit=True, select_stns=[]):
        """Return a dict of short traces from a tag/trcnum based on picks. Omits un-picked traces by default."""
        if not tag:
            event_str,tag,trace_num = self.get_event(event_id)
        else:
            event_str = utils.parse_eid(event_id)
        traces = {}
        picks = self.get_event_picks(event_id)
        o_ind = self.get_event_origin_index(event_id)
        if pre == -1: # don't care about picks at all
            sl = slice(o_ind,o_ind+tot_len)
            if not select_stns: select_stns = self.stns
            for stn in select_stns:
                traces[stn] = self.waveforms["L0_" + stn][tag][trace_num].data[sl]
        else:
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
                        pp = int(travel/2.74 * 40) + o_ind
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
        if 'LabPicks' in self.auxiliary_data.list():
            return self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"][event_str].parameters
        else:
            return self.auxiliary_data.LabEvents[event_str].picks.parameters
    
    def dec_plot_all(self, tag, trc_num, beg=0, cut=None, figname='temp.html'):
        """Show all stations on an interactive plot. Decimated by 20 to handle full trace lengths.
        """
        if not cut:
            stn = 'AE17'
            cut = len(self.waveforms['L0_'+stn][tag][trc_num].data)
        fig = go.Figure()
        for stn in self.stns:
            dctr = signal.decimate(self.waveforms['L0_'+stn][tag][trc_num].data[beg:cut],20)
            x = np.arange(beg,cut,20)
            fig.add_trace(go.Scattergl(x=x,y=dctr,name=stn))
        fig.write_html(figname)
        print(f'Plotted {os.getcwd()+figname}')

    def update_event_index(self):
        ed = {}
        for ev in self.auxiliary_data.LabEvents.list():
            lev = self.auxiliary_data.LabEvents[ev]
            tag = lev.info.parameters['tag']
            if tag not in ed.keys():
                ed[tag] = {}
            tr = f"tr{lev.info.parameters['trace']}"
            if tr not in ed[tag].keys():
                ed[tag][tr] = [ev]
            else:
                ed[tag][tr].append(ev)
                
        # save as aux data
        if 'EventIndex' in self.auxiliary_data.list():
            del self.auxiliary_data.EventIndex
        for tag in ed.keys():
            self.add_auxiliary_data(data=np.array([]),
                                    data_type='EventIndex',
                                    path=tag,
                                    parameters=ed[tag])

######## picking helpers ########
# TODO move these to a picking module

def pick_near(trace, near, reach_left=2000, reach_right=2000, thr=0.9, AIC=[], trace_start=0):
    """Get a list of picks for one trace."""
    if len(AIC) == 0:
        AIC = get_AIC(trace, near, reach_left, reach_right)
    picks = (
        peakutils.indexes(AIC[5:-5] * (-1), thres=thr, min_dist=50) + 5
    )  # thres works better without zeros
    picks = list(picks + (near - reach_left) + trace_start)  # window indices -> trace indices
    # TODO: Do I need to remove duplicates and sort?
    return picks


def get_AIC(trace, near, reach_left=2000, reach_right=2000):
    """Calculate the AIC function used for picking. Accepts an Obspy trace or array-like data."""
    if hasattr(trace, "data"):
        window = trace.data[near - reach_left : near + reach_right]
    else:
        window = trace[near - reach_left : near + reach_right]
    return AIC_func(window)

def AIC_func(data,omit=5):
    AIC = np.zeros_like(data)
    for i in range(omit, len(data)-omit):
        AIC[i] = i * np.log10(np.var(data[:i])) + (len(data) - i) * np.log10(
            np.var(data[i:]))
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
