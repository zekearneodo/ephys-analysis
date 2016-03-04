import sys, os
import glob
import numpy as np
import h5py as h5
import pandas as pd

def get_kwik(block_path):
    '''
    Returns the kwik file found in the block path
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    kwik : full path name to kwik file
    '''
    kwik = glob.glob(os.path.join(block_path,'*.kwik'))
    assert len(kwik)==1
    return kwik[0]

def get_kwd(block_path):
    '''
    Returns the raw.kwd file found in the block path
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    kwd : full path name to raw.kwd file
    '''
    kwd = glob.glob(os.path.join(block_path,'*.raw.kwd'))
    assert len(kwd)==1
    return kwd[0]

def get_prb(block_path):
    '''
    Returns the *.prb file found in the block path
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    prb : full path name to *.prb file
    '''
    ls = glob.glob(os.path.join(block_path,'*.prb'))
    assert len(ls)==1
    return ls[0]

# def csv(block,name):
#     return os.path.join(EXP_DIR,TABLE_DIR,'{}_{}.csv'.format(block,name))

## top-level functions to read data from kwik files

def read_events(block_path,event_type):
    '''
    Reads events from the kwik file associated with a block

    example use:

    >>> digmarks = read_events(block_path,'DigMark')
    >>> stimulus = read_events(block_path,'Stimulus')
    
    Parameters
    ------
    block_path : str
        path to the block
    event_type : str
        the type of event
    
    Returns
    ------
    events : pandas DataFrame
    
    '''
    with h5.File(get_kwik(block_path),'r') as kf:
        events = {}
        for col in kf['/event_types'][event_type]:
            events[col] = kf['/event_types'][event_type][col][:]
    return pd.DataFrame(events)

def get_fs(block_path):
    '''
    Reads sampling rate in Hz from the kwik file associated with a block
    
    Parameters
    ------
    block_path : str
        path to the block
    event_type : str
        the type of event
    
    Returns
    ------
    events : pandas DataFrame
    
    '''
    with h5.File(get_kwik(block_path),'r') as kf:
        fs = kf['/recordings/0'].attrs['sample_rate']
    return fs


QUAL_LOOKUP = {0: 'Noise',
               1: 'MUA',
               2: 'Good',
               3: 'unsorted',
              }


def get_qual(block_path,cluster):
    '''
    Returns labeled cluster quality ('Noise', 'MUA', 'Good', 'unsorted')
        for a given cluster
    
    Parameters
    ------
    block_path : str
        path to the block
    cluster : int
        the cluster id
    
    Returns
    ------
    quality : string
        one of the following: ('Noise', 'MUA', 'Good', 'unsorted')
    
    '''
    with h5.File(get_kwik(block_path),'r') as kf:
        qual = kf['/channel_groups/0/clusters/main/']["%i" % cluster].attrs['cluster_group']
    return QUAL_LOOKUP[qual]

def get_clusters(block_path,channel_group=0,clustering='main'):
    '''
    Returns a dataframe of clusters observed in kwik file
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    clusters : pandas DataFrame
    
    '''    
    with h5.File(get_kwik(block_path),'r') as kf:
        observed_clusters = np.unique(
            kf['/channel_groups/{}/spikes/clusters/{}'.format(channel_group,clustering)][:]
            )
        clusters = pd.DataFrame({'cluster': observed_clusters})
        clusters['quality'] = clusters['cluster'].map(lambda clu: get_qual(block_path,clu))
    return clusters

def get_spikes(block_path,channel_group=0,clustering='main'):
    '''
    Returns a dataframe of spikes observed in kwik file
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    spikes : pandas DataFrame
    
    '''    
    with h5.File(get_kwik(block_path),'r') as kf:
        spikes = pd.DataFrame(
            dict(cluster=kf['/channel_groups/{}/spikes/clusters/{}'.format(channel_group,clustering)][:],
                 recording=kf['/channel_groups/{}/spikes/recording'.format(channel_group)][:],
                 time_samples=kf['/channel_groups/{}/spikes/time_samples'.format(channel_group)][:],
                 )
            )
    return spikes
