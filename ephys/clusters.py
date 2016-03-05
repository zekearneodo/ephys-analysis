import os
import glob
import numpy as np
from scipy.interpolate import UnivariateSpline
from core import ls, load_probe, get_fs

@ls
def find_mean_waveforms(block_path,cluster,cluster_store=0,clustering='main'):
    return os.path.join(block_path,
                        '*.phy',
                        'cluster_store',
                        str(cluster_store),
                        clustering,
                        '{}.mean_waveforms'.format(cluster)
                        )

@ls
def find_mean_masks(block_path,cluster,cluster_store=0,clustering='main'):
    return os.path.join(block_path,
                        '*.phy',
                        'cluster_store',
                        str(cluster_store),
                        clustering,
                        '{}.mean_masks'.format(cluster)
                        )

def mean_masks_w(block_path,clu):
    '''
    Weights are equivalent to the mean_mask values for the channel.
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    
    Returns
    ------
    w : weight vector
    '''
    mean_masks = find_mean_masks(block_path,clu)
    mean_masks_arr = np.fromfile(mean_masks,dtype=np.float32)
    return mean_masks_arr

def max_masks_w(block_path,clu):
    '''
    Places all weight on the channel(s) which have the largest mean mask values.

    If more than one channel have a mean_mask value equal to the max, these
    channels will be weighted equally.
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    
    Returns
    ------
    w : weight vector
    '''
    w = mean_masks_w(block_path,clu)
    return w==w.max()


def get_cluster_coords(block_path,clu,weight_func=None):
    '''
    Returns the location of a given cluster on the probe in x,y coordinates
        in whatever units and reference the probe file uses. 
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    weight_func : function
        function which takes `block_path` and `clu` as args and returns a weight
        vector for the coordinates. default: max_masks_w 
    
    Returns
    ------
    xy : numpy array of coordinates
    '''
    if weight_func is None:
        weight_func = max_masks_w
    w = weight_func(block,clu)

    prb_info = load_probe(block_path)
    channels = prb_info.channel_groups[0]['channels']
    geometry = prb_info.channel_groups[0]['geometry']
    coords = np.array([geometry[ch] for ch in channels])    
    
    return np.dot(w,coords) / w.sum()



## spike shapes

def upsample_spike(spike_shape,fs,new_fs=200.0):
    '''
    upsamples a spike shape to prepare it for computing the spike width 
    
    Parameters
    ------
    spike_shape : numpy array
        the spike shape
    fs : float
        the sampling rate of the spike shape
    new_fs : float
        sampling rate to upsample to (default=200Hz)
    
    Returns
    ------
    time : numpy array
        array of sample times in seconds
    new_spike_shape :
        upsampled spike shape
    '''
    t = np.arange(0,spike_shape.shape[0]/fs,1/fs)[:spike_shape.shape[0]]
    spl = UnivariateSpline(t,spike_shape)
    ts = np.arange(0,len(spike_shape)/fs,1/new_fs)
    return ts, spl(ts)
    
def get_troughpeak(time,spike_shape):
    '''
    grabs the time of the trough and peak
    
    Parameters
    ------
    time : numpy array
    spike_shape : numpy array
    
    Returns
    ------
    trough_time : float
        time of trough in seconds
    peak_time : float
        time of peak in seconds
    '''
    trough_i = spike_shape.argmin()
    peak_i = spike_shape[trough_i:].argmax()+trough_i
    return time[trough_i],time[peak_i]

def get_width(block_path,clu,new_fs=200.0):
    '''
    grabs the time of the trough and peak
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    
    Returns
    ------
    width : float
        the width of the spike in seconds
    '''
    fs = get_fs(block_path)
    exemplar = get_spike_exemplar(block_path,clu)
    
    trough,peak = get_troughpeak(*upsample_spike(exemplar,fs,new_fs=new_fs))

    return peak-trough


def get_spike_exemplar(block_path,clu):
    '''
    returns an exemplar of the spike shape
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    
    Returns
    ------
    exemplar : float
        the width of the spike in seconds
    '''
    prb_info = load_probe(block_path)
    
    mean_waveform = find_mean_waveforms(block,clu)
    arr = np.fromfile(mean_waveform,dtype=np.float32).reshape((-1,len(prb_info.channel_groups[0]['channels'])))
    
    mean_masks = find_mean_masks(block,clu)
    mean_masks_arr = np.fromfile(mean_masks,dtype=np.float32)
    
    return arr[:,mean_masks_arr.argmax()]
