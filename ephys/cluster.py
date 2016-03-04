
from scipy.interpolate import UnivariateSpline

## cluster metrics

def find_mean_waveforms(block_path,cluster):
    return os.path.join(EXP_DIR,KLUSTA_DIR,block,block.split('__')[-1]+'.phy',
                        'cluster_store','0','main','{}.mean_waveforms'.format(cluster))
    
def find_mean_masks(block_path,cluster):
    return os.path.join(EXP_DIR,KLUSTA_DIR,block,block.split('__')[-1]+'.phy',
                        'cluster_store','0','main','{}.mean_masks'.format(cluster))


def get_cluster_loc(mean_masks,prb,weighted=False):
    prb_info = imp.load_source('prb',prb)
    
    channels = prb_info.channel_groups[0]['channels']
    geometry = prb_info.channel_groups[0]['geometry']

    coords = np.array([geometry[ch] for ch in channels])
    
    mean_masks_arr = np.fromfile(mean_masks,dtype=np.float32)

    if weighted:
        return np.dot(mean_masks_arr,coords) / mean_masks_arr.sum()
    else:
        return coords[mean_masks_arr.argmax(),:]

def plot_mean_waveform(mean_waveform,prb,chan_alpha=None,scale_factor=0.05,color='0.5',**plot_kwargs):
    prb_info = imp.load_source('prb',prb)
    
    channels = prb_info.channel_groups[0]['channels']
    geometry = prb_info.channel_groups[0]['geometry']

    coords = np.array([geometry[ch] for ch in channels])
    
    arr = np.fromfile(mean_waveform,dtype=np.float32).reshape((-1,len(channels)))
    
    if chan_alpha is None:
        chan_alpha = np.ones(len(coords))
    
    for waveform,xy,alpha in zip(arr.T,coords,chan_alpha):
#         print mask
        plt.plot(xy[0]+np.arange(len(waveform))-len(waveform)/2,
                 waveform*scale_factor+xy[1],
                 color=color,
                 alpha=alpha,
                 **plot_kwargs
                )
        try:
            label = plot_kwargs.pop('label')
        except KeyError:
            pass

# 
def get_cluster_coords(block_path,clu,weighted=False):
    '''
    Returns the location of a given cluster on the probe in x,y coordinates
        in whatever units and reference the probe file uses. 
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    weighted : boolean
        whether to weight across channels or only find primary
    
    Returns
    ------
    xy : numpy array of coordinates
    '''
    mean_masks = find_mean_masks(block_path,clu)
    prb = find_prb(block_path)
    return get_cluster_loc(mean_masks,prb,weighted=weighted)

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
    raw_shape = get_raw_shape(block_path,clu)
    
    trough,peak = get_troughpeak(*upsample_spike(raw_shape,fs,new_fs=new_fs))

    return peak-trough


def get_raw_shape(block,clu):
    prb = find_prb(block)
    prb_info = imp.load_source('prb',prb)
    
    mean_waveform = find_mean_waveforms(block,clu)
    arr = np.fromfile(mean_waveform,dtype=np.float32).reshape((-1,len(prb_info.channel_groups[0]['channels'])))
    
    mean_masks = find_mean_masks(block,clu)
    mean_masks_arr = np.fromfile(mean_masks,dtype=np.float32)
    
    return arr[:,mean_masks_arr.argmax()]
