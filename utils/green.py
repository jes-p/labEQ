"""Module for managing Green's functions."""
import matlab.engine
import h5py

def get_greens(dists, post_len, vp=2.74, vs=1.4, source_type='moment',path = '/home/jes/data/Lab_Data/BP3_GF/'):
    """Version 0.0.0, highly unstable. Currently gets or makes GF for BP3 at 40Mhz only.
    Get or compute greens functions. Uses parallel processing in matlab to compute new GF. Start a matlab session and parallel pool separately, then call matlab.engine.ShareEngine to save ~20s per call of this function."""
    
    green = {}
    calc_dists = []
    path += source_type + '/'
    for d in dists:
        # check for existing GF.mat files for each distance
        d_split = str(d).split('.')
        if int(d_split[1]) == 0: d_split[1] = '' # all .00... outputs are the same
        fname = f'{d_split[0]}_{d_split[1]}.mat'
        try:
            with h5py.File(path+fname,'r') as f:
                # check length
                if f['Green']['post_len'] < post_len: raise EOFError('Not enough points in file')
                # if still here then long enough so read it in
                gdict = {}
                for k,v in f['Green'][source_type[0]]:
                    gdict[k] = v
                green[dist] = gdict
            
        except:
            # add distance to list of dists to compute
            calc_dists.append(dist)
            
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
            eval_string = f'mat_{source_type}_GF(cp,cs,dists,post_len,"{path}")'
            done = eng.eval(eval_string)
    
    # make sure the function waited for matlab to finish
    assert done == 1
    
    # read in newly calculated GF using h5py
    # re-organize so this isn't repeated code
    for d in calc_dists:
        d_split = str(d).split('.')
        if int(d_split[1]) == 0: d_split[1] = '' # all .00... outputs are the same
        fname = f'{d_split[0]}_{d_split[1].mat}'
        with h5py.File(path+fname,'r') as f:
            gdict = {}
            for k,v in f['Green'][source_type[0]]:
                gdict[k] = v
            green[dist] = gdict
     
    return green