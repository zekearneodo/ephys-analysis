
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
    