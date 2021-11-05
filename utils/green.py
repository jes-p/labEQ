"""Module for managing Green's functions."""
import matlab.engine
import h5py
import numpy as np
from os.path import exists as path_exists

def get_greens(dist_dict, post_len, pre=200, vp=2.74, vs=1.4, dist_prec=2, source_type='moment',path = '/home/jes/data/Lab_Data/BP3_GF/',verbose=1):
    """Version 0.0.0, highly unstable. Currently gets or makes GF for BP3 at 40Mhz only.
    Get or compute greens functions. Uses parallel processing in matlab to compute new GF. Start a matlab session and parallel pool separately, then call matlab.engine.ShareEngine to save ~20s per call of this function."""
    
    green = {}
    calc_dists = []
    fullpath = path + source_type + '/'
    for d in dist_dict.values():
        if not check_for_green(d, post_len, dist_prec, source_type, path):
            calc_dists.append(d)
           
    # calculate any new (or too short) dists
    calc_greens(calc_dists, post_len, vp=vp, vs=vs, dist_prec=dist_prec, source_type=source_type, path=path, verbose=verbose)
    
    # read in GF using h5py
    for s,d in dist_dict.items():
        fname = get_fname(d,dist_prec)
        with h5py.File(fullpath+fname,'r') as f:
            gdict = {}
            # get first arrival from raytime
            arr = round(f['Green']['raytime'][:].T[0,-1] * 40e6)
            for k in f['Green'][source_type[0]].keys():
                gdict[k] = np.squeeze(f['Green'][source_type[0]][k][arr-pre:arr+post_len])
            green[s] = gdict
     
    return green

def get_raytime(dist, post_len, pre=200, vp=2.74, vs=1.4, dist_prec=2, source_type='moment', path = '/home/jes/data/Lab_Data/BP3_GF/', verbose=1):
    """Get raytime matrix for one distance.
    """
    fullpath = path + source_type + '/'
    if not check_for_green(dist, post_len, dist_prec, source_type, path):
        raise ValueError('GF not calculated for these parameters.')
    fname = get_fname(dist,dist_prec)
    with h5py.File(fullpath+fname,'r') as f:
        return f['Green']['raytime'][:].T


def calc_greens(dist_list: list, post_len, vp=2.74, vs=1.4, dist_prec=2, source_type='moment',path = '/home/jes/data/Lab_Data/BP3_GF/',verbose=1):
    """Calculate greens functions for a list of distances.
    """
    calc_dists = []
    fullpath = path + source_type + '/'
    for d in dist_list:
        # check for existing GF.mat files for each distance
        fname = get_fname(d,dist_prec)
        if not check_for_green(d, post_len, dist_prec, source_type, path):
            # add distance to list of dists to compute
            calc_dists.append(d)
            
    # calculate any new (or too short) dists
    if len(calc_dists) > 0:
        with matlab.engine.connect_matlab() as eng:
            eng.eval("addpath('/home/jes/dev/PlateSoln')")
            # set matlab variables for GF calc (can't just f-str them in)
            eng.workspace['dists'] = matlab.double([float(d) for d in calc_dists]) # dists in mm, import lists using the matlab.double constructor
            eng.workspace['cp'] = float(vp)
            eng.workspace['cs'] = float(vs)
            eng.workspace['post_len'] = float(post_len)
            # calculate all GF
            eval_string = f'mat_{source_type}_GF(cp,cs,dists,post_len,"{fullpath}",{verbose})'
            done = eng.eval(eval_string)
    else:
        done = 1
    
    # make sure the function waited for matlab to finish (is this necessary?)
    assert done == 1
    
    # just calculating, nothing to return
    return

def check_for_green(dist, post_len, dist_prec=2, source_type='moment',path = '/home/jes/data/Lab_Data/BP3_GF/'):
    """Check if the requested greens function file exists and has enough points. Returns False if the file does not exist/is insufficient length.
    """
    skip = False
    path += source_type + '/'
    fname = get_fname(dist,dist_prec)
    if path_exists(path+fname):
            with h5py.File(path+fname,'r') as f:
                # check length
                if f['Green']['post_len'][0] >= post_len:
                    skip = True
    return skip
         
    
    
    
def get_fname(dist, dist_prec=2):
    """Get the filename for a greens function distance.
    """
    d = np.round(dist,dist_prec)
    a,b = str(d).split('.')
    # drop all trailing zeros after the decimal
    while b!='' and b[-1] == '0':
        b = b[:-1]
    fname = f'{a}_{b}.mat'
    return fname