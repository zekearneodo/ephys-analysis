
import seaborn as sns


def plot_cluster(block_path,clu,color='0.5',**plot_kwargs):
    '''
    Returns the location of a given cluster on the probe in x,y coordinates
        in whatever units and reference the probe file uses. 
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    color
        whether to weight across channels or only find primary
    
    '''
    mean_waveforms = find_mean_waveforms(block,clu)
    mean_masks_arr = np.fromfile(find_mean_masks(block,clu),dtype=np.float32)
    prb = find_prb(block)

    plot_mean_waveform(mean_waveforms,prb,chan_alpha=mean_masks_arr,color=color,**plot_kwargs)
    

def plot_all_clusters(block,clusters=None):
    
    if clusters is None:
        clusters = get_clusters(block)
        clusters = clusters[clusters.quality.isin(['Good','MUA'])].sort_values(['quality','cluster']).reset_index()

    palette = sns.color_palette("hls", len(clusters))

    f = plt.figure(figsize=(6, 12))
    for idx,cluster_row in clusters.iterrows():

        clu = cluster_row.cluster
        qual = cluster_row.quality

        plot_cluster(block,clu,color=palette[idx],scale_factor=0.05,label="{}({})".format(clu,qual))

    plt.axis('equal')
    leg = plt.legend(loc='center left')
    [plt.setp(label, alpha = 1.0) for label in leg.get_lines()]
    sns.despine(bottom=True,left=True)
    plt.xticks([])
    plt.yticks([])



def plot_spike_shape(block,clu,normalize=False,**kwargs):
    fs = get_fs(block)
    raw_shape = get_raw_shape(block,clu)
    
    time,shape = upsample_spike(raw_shape,fs)
    time -= time[shape.argmin()]
    
    if normalize==True:
        shape /= -shape[shape.argmin()]
    
    plt.plot(time,shape,**kwargs)
    


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
