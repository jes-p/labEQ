"""Module for managing Green's functions."""
import matlab.engine
import h5py

def get_greens(dists, post_len, vp=2.74, vs=1.4):
    """Version 0.0.0, highly unstable. Currently gets or makes GF for BP3 at 40Mhz only."""
    
    green = {}
    calc_dists = []
    for d in dists:
        try:
            # check for existing GF.mat files for each distance
            d_split = str(d).split('.')
            fname = f'{d_split[0]}_{d_split[1].mat}'
            with h5py.File(fname,'r') as f:
                # check length
                if f['Green']['post_len'] < post_len: raise EOFError('Not enough points in file')
                # if still here then long enough so read it in
                mdict = {}
                for k,v in f['Green']['m']:
                    mdict[k] = v
                green[dist] = mdict
            
        except:
            # add distance to list of dists to compute
            calc_dists.append(dist)
            
    # calculate any new (or too short) dists
    if len(calc_dists) > 0:
        with matlab.engine.start_engine() as eng:
            eng.eval("addpath('/home/jes/Dropbox/Glaser_Lab/PlateSoln')")
            # set matlab variables for GF calc (can't just f-str them in)
            eng.workspace['dists'] = matlab.double([float(d) for d in calc_dists]) # dists in mm, import lists using the matlab.double constructor
            eng.workspace['cp'] = float(vp)
            eng.workspace['cs'] = float(vs)
            eng.workspace['post_len'] = float(post_len)
            # calculate all GF
            done = eng.eval('mat_moment_GF(cp,cs,dists,post_len,"/home/jes/data/Lab_Data/BP3_GF/moment/")')
    
    # make sure the function waited for matlab to finish
    assert done == 1
    
    # read in newly calculated GF using h5py
    # re-organize so this isn't repeated code
    for d in calc_dists:
        d_split = str(d).split('.')
        fname = f'{d_split[0]}_{d_split[1].mat}'
        with h5py.File(fname,'r') as f:
            mdict = {}
            for k,v in f['Green']['m']:
                mdict[k] = v
            green[dist] = mdict
     
    return green