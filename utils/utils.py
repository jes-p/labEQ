"Additional utils for labEQ"
import numpy as np
import os
from scipy import signal

# Import plotting libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import erfc for fspec_coeff
from scipy.special import erfc

def parse_eid(event_id):
    """Parse an event_str or integer event_num input into the event_str"""
    if isinstance(event_id,int):
        event_str = f'event_{event_id:0>3d}'
    else:
        event_str = event_id
    return event_str

def vr(base,model):
    """Return VR in percent for model compared to base.
    """
    return (1-np.sum(np.power(base-model,2))/np.sum(np.power(base,2)))*100

def xcorr_shift(base,model,scale=False,partial=None):
    """ UPDATE: return shifted (and optionally scaled) version of model
    Get offset and scale factor between base and model based on cross-correlation. Lag is given as model compared to base. Positive output indicates that the model arrives late. E.g. lag = 12 will be aligned by slicing model[12:].
    Scale factor is also model compared to base. Multiply the model by the scale factor for the best fit.
    """
    if not partial:
        partial = len(base)
    acorr = signal.correlate(base[:partial], base[:partial], mode='same')
    corr = signal.correlate(model[:partial], base[:partial], mode='same')
    lags = signal.correlation_lags(len(model[:partial]), len(base[:partial]), mode='same')
    shift = lags[np.argmax(corr)]
    pad = np.zeros(np.abs(shift))
    fac = np.max(acorr)/np.max(corr)
    if not scale: fac = 1
    if shift >= 0:
        out = np.concatenate((fac*model[shift:],pad))
    else:
        out = np.concatenate((pad,fac*model[:shift]))
    # return lags[np.argmax(corr)], np.max(acorr)/np.max(corr)
    return out

def xcorr_coeff(base,model):
    """Get the correlation coefficient for the model compared to base, with zero shift.
    """
    l = len(base)
    if len(model) != l:
        raise ValueError("Model and base must be the same length")
    base_acorr = signal.correlate(base, base)[l-1]
    model_acorr = signal.correlate(model, model)[l-1]
    xcorr = signal.correlate(model, base)[l-1]
    return xcorr**2/(base_acorr*model_acorr) * 100

def fspec_coeff(base,model,smooth=10,fmax=None,dt=1e-7,log=False):
    """Get an amplitude-based fit coefficient for the model compared to base. The fourier spectra are smoothed by a moving average and only compared up to fmax.
    """
    nf = len(base)
    if len(model) != nf: raise ValueError('Model and base must be the same length')
    bspec = np.abs(np.fft.rfft(base)[:-1])
    mspec = np.abs(np.fft.rfft(model)[:-1])
    ff = np.fft.fftfreq(nf,d=dt)[:nf//2]
    # smoothing
    box = np.ones(smooth)/smooth
    sbs = np.convolve(bspec,box,mode='same')
    sms = np.convolve(mspec,box,mode='same')
    if log:
        sbs = np.log10(sbs)
        sms = np.log10(sms)
    snr = 2 * np.abs((sbs-sms)/(sbs+sms)) # normalized residuals
    if fmax:
        cut = np.argwhere(ff>fmax)[0,0]-1
    else:
        cut = len(ff)
    score = 100 * np.sum(erfc(snr[:cut]))/cut
    return score
    
def calc_sptime(dist,vp=2.74,vs=1.4,output='points'):
    """Find the theoretical S-P time for an epicentral distance, in mm.
    Output in sample points (points) or microseconds (us).
    """
    direct = np.sqrt(dist**2+38.5**2)
    p_arr = direct/vp
    s_arr = direct/vs
    if output == 'points':
        return round((s_arr-p_arr)*40)
    elif output == 'us':
        return s_arr-p_arr
    else:
        raise InputError('Invalid output option. Choose points or us.')

######## source helpers ########
def ball_force(
    diam=(1.18e-3), rho=7850, nu=[0.28, 0.3], E=[200e9, 6e9], h=0.305, Fs=40e6
):
    """
    calculate the force function from a ball drop
    radius in m, ball density in kg/m^3, ball and surface PR, ball and surface YM in Pa
    drop height in m, sampling freq. in Hz
    pmma defaults: 1190, .3-.34, 6.2e9
    steel: 7850, .28, 214e9"""
    radius = diam / 2
    v = np.sqrt(2 * 9.8 * h)
    d = sum((1 - np.array(nu)) / (np.pi * np.array(E)))
    tc = 4.53 * (4 * rho * np.pi * d / 3) ** 0.4 * radius * v ** -0.2
    fmax = 1.917 * rho ** 0.6 * d ** -0.4 * radius ** 2 * v ** 1.2
    ftime = np.arange(0, tc * 1.01, 1 / Fs)
    ffunc = -1 * np.nan_to_num(
        fmax * np.power(np.sin(np.pi * ftime / tc), 1.5)
    )  # times past tc are nan->0
    return tc, ffunc

def get_phi(stn_loc,src_loc):
    """Get azimuth (phi) in radians for station and source coordinates. The top block motion is in the -x direction.
    """
    dx,dy = [stn_loc[i]-src_loc[i] for i in [0,1]]
    if dx==0:
        phi = (np.pi/2) if dy>0 else (-1*np.pi/2)
    elif dx<0:
        phi = np.arctan(dy/dx)+np.pi
    else:
        phi = np.arctan(dy/dx)
    return phi

######## plotting helpers ########

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

def grid_plot(plot_data: dict, plot_file_name, legend=True, number_plots=False, plot_titles=False, fig_title=False):
    """Make and save a 4x4 plotly plot. Takes a dict like {plot title: [items to plot]} and file name to save the figure. Items should be a list of dicts of keyword args accepted by go.Scattergl. Doesn't return the figure.
    Useful Scattergl keywords:
        y: required, no default (all others optional)
        x: defaults to indices
        name: legend entry for trace
        legendgroup: groups legend items to toggle together (e.g. 'Raw', 'Preprocess', 'New Picks', etc.)
        """
    if not plot_titles:
        if number_plots:
            titles = [stn+f" ({i})" for i,stn in enumerate(plot_data.keys())]
        else:
            titles = plot_data.keys()
    else:
        titles = plot_titles
    fig,plotkey = subplts(4,4,titles)
    color_list = ['black', 'red', 'blue', 'green', 'orange', 'magenta', 'gray']
    ncol = len(color_list)
    for i,ttl in enumerate(plot_data.keys()): 
        # iterate through traces
        for j,tdict in enumerate(plot_data[ttl]):
            # set default values
            linecolor = {'color': color_list[j % ncol]}
            if 'line' in tdict.keys():
                tdict['line'].update(linecolor)
            else:
                tdict.update({'line':linecolor})
            # default to drop legends after first plot for neatness
            if i > 0:
                tdict.update({'showlegend':False})
            # add trace to plot
            fig.append_trace(go.Scattergl(**tdict),int(plotkey[i][0]),plotkey[i][1])
            
    # finish figure
    if not legend:
        fig["layout"].update(showlegend=False)
    if fig_title:
        fig.update_layout(title=fig_title)
    fig.update_annotations(font_size=12)
    fig.write_html(plot_file_name + '.html')
    if plot_file_name[0] == '/':
        pdir = ''
    else:
        pdir = os.getcwd() + '/'
    print(f'Plotted {pdir+plot_file_name}.html')
