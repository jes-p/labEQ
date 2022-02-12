"""Module to manage green's functions (calculated with CPS)."""

import subprocess
import obspy
import warnings

def get_greens(dists,test_dir):
    """Get impulse response GF for the dists dict from the test_dir.
    Will be updated with a default/deprecated test_dir input after tuning CPS parameters.
    Returns gf as dict with stns from dists as keys. Each entry contains:
    - step: step response (diff to get impulse response)
    - b: beginning of trace (s), relative to event origin
    - a: direct P arrival (s)
    - t0: direct S arrival (s)
    - dt: sample step size (s) (inverse of Fs (Hz))
    """
    gf = {}
    for stn,d in dists.items():
        try:
            strm = obspy.read(f'{test_dir}/{d*100:06.0f}3850.ZVF')
            gf[stn] = {'step':strm[0].data, 'b':strm[0].stats.sac['b'],
                       'a':strm[0].stats.sac['a'], 't0':strm[0].stats.sac['t0'],
                       'dt':strm[0].stats.sac['delta']}
        except FileNotFoundError:
            warnings.warn(f'GF not computed for distance {d:.2f}mm',stacklevel=2)
    return gf

def run_cps(test_dir, dt, npts, dists, vp = 2.74, vs = 1.4, qp = 4e3, qs = 1e3, frefp = 1e7, frefs = 1e7, etap=0.0, etas=0.0, vred=0, t0=0, qk=False):
    """Run CPS (hspec96) for the 38.5mm base plate in the name test_dir.
    Dists in mm, velocities in mm/us, all others in s, Hz, etc.
    """
    subprocess.run(['mkdir', '-p', test_dir])

    # write dist file
    write_cps_dfile(test_dir, dists, dt, npts, vred=vred, t0=t0)
    
    # write modfile
    write_cps_model(test_dir, vp=vp, vs=vs, qp=qp, qs=qs, frefp=frefp, frefs=frefs, ep=etap, es=etas)

    # write run script
    write_cps_script(test_dir, qk=qk)

    # run it
    subprocess.run(f'{test_dir}/run_fk',cwd=test_dir)

def write_cps_dfile(test_dir, dists, dt, npts, vred=0, t0=0):
    """Write dist file for CPS run in test_dir.
    Inputs:
        - test_dir: full path to current test (str)
        - dists: dict like {stn: d in mm} or list [d in mm]
        - dt: sample spacing in s
        - npts: int
        - vred: reduction velocity in mm/us
        - t0: amount to shift start from vred-based first arrival in s, likely negative
        """
    if isinstance(dists,dict):
        dist_list = dists.values()
    elif isinstance(dists,list):
        dist_list = dists
    else:
        raise ValueError('dists not a dict or list')
        
    with open(test_dir+'/dfile','w') as f:
        for d in dist_list:
    #         r = np.sqrt(38.5**2+d**2)
    #         us = r/1.4 + 5
    #         npts = np.ceil(us/(1e6*dt))
    #         if npts > max_pts: continue
            # write dfile lines with modeling units: cm and s
            f.write(f'{d:.2f}E-01\t{dt:.1E}\t{npts:n}\t{t0:.1E}\t{vred/10:.2E}\n')
            

def write_cps_script(test_dir,h=38.5,pulse='p1', qk=False):
    cps_bin = '/home/jes/dev/PROGRAMS.330/bin/'
    pt = pulse[0]
    l = pulse[1:]
    kflag = '-K'
    with open(test_dir+'/run_fk','w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('# prep\n')
        f.write(f'{cps_bin}hprep96 -M modfile -d dfile -HS {h/10:.2E} -TF -BF -ALL\n\n')
        f.write('# run\n')
        f.write(f'{cps_bin}hspec96 {qk*kflag} -SU > hspec96.out\n\n')
        f.write('# pulse\n')
        f.write(f'{cps_bin}hpulse96 -{pt} -l {l} -D | {cps_bin}f96tosac -E\n')
    subprocess.run(['chmod', '+x', f'{test_dir}/run_fk'])
    
def write_cps_model(test_dir, h=38.5, vp=2.74, vs=1.4, rho=1.13, qp=4e3, qs=1e3, ep=0, es=0, frefp=1e7, frefs=1e7):
    with open(test_dir+'/modfile','w') as f:
        f.writelines([l+'\n' for l in ['MODEL.01', 'test plate in real units', 'ISOTROPIC', 'CGS',
                      'FLAT EARTH', '1-D', 'CONSTANT VELOCITY',
                      'LINE08', 'LINE09', 'LINE10', 'LINE11',
                      '\tH(CM)\tVP(CM/S)\tVS(CM/S)\tRHO(GM/CC)\tQP\tQS\tETAP\tETAS\tFREFP\tFREFS\n']])
        f.write(f'\t{h/10:.2E}\t{vp*1e5:.2E}\t{vs*1e5:.2E}\t{rho:.3f}\t'+
                f'{qp:.2E}\t{qs:.2E}\t{ep:.2f}\t{es:.2f}\t{frefp:.1E}\t{frefs:.1E}\n')