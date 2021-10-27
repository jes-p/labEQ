"""Module for managing Green's functions."""
import matlab.engine
import h5py
import numpy as np

def get_greens(dists, post_len, vp=2.74, vs=1.4, dist_prec=2, source_type='moment',path = '/home/jes/data/Lab_Data/BP3_GF/',verbose=1):
    """Version 0.0.0, highly unstable. Currently gets or makes GF for BP3 at 40Mhz only.
    Get or compute greens functions. Uses parallel processing in matlab to compute new GF. Start a matlab session and parallel pool separately, then call matlab.engine.ShareEngine to save ~20s per call of this function."""
    
    green = {}
    calc_dists = []
    path += source_type + '/'
    for d in dists:
        # check for existing GF.mat files for each distance
        d = np.round(d,dist_prec)
        a,b = str(d).split('.')
        # drop all trailing zeros after the decimal
        while b!='' and b[-1] == '0':
            b = b[:-1]
        fname = f'{a}_{b}.mat'
        try:
            with h5py.File(path+fname,'r') as f:
                # check length
                if f['Green']['post_len'][0] < post_len: raise EOFError('Not enough points in file')
                # if still here then long enough so read it in
                gdict = {}
                for k,v in f['Green'][source_type[0]].items():
                    gdict[k] = v
                green[d] = gdict
            
        except:
            # add distance to list of dists to compute
            print(f'Check failed for {d}')
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
            eval_string = f'mat_{source_type}_GF(cp,cs,dists,post_len,"{path}",{verbose})'
            done = eng.eval(eval_string)
    else:
        done = 1
    
    # make sure the function waited for matlab to finish
    assert done == 1
    
    # read in newly calculated GF using h5py
    # re-organize so this isn't repeated code
    for d in calc_dists:
        a,b = str(d).split('.')
        # drop all trailing zeros after the decimal
        while b!='' and b[-1] == '0':
            b = b[:-1]
        fname = f'{a}_{b}.mat'
        with h5py.File(path+fname,'r') as f:
            gdict = {}
            for k,v in f['Green'][source_type[0]].items():
                gdict[k] = v
            green[d] = gdict
     
    return green