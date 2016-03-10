import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import core

def kwik2rigid_pandas(block_path):
    spikes = core.get_spikes(block_path)
    stims = get_acute_stims(block_path)
    spikes['stim_name'], spikes['stim_presentation'], spikes['stim_time_stamp'] = zip(*spikes["time_samples"].map(_EventAligner(stims).stim_checker))
    spikes['stim_aligned_time_stamp'] = spikes['time_samples'].values.astype('int') - spikes['stim_time_stamp'].values
    del spikes['time_samples']
    del spikes['stim_time_stamp']
    timestamp2time(spikes, core.get_fs(block_path), 'stim_aligned_time_stamp', 'stim_aligned_time')
    return spikes, stims

def get_acute_stims(block_path):
    '''
    Fast code to load up stimuli information for an acute recording

    Parameters
    -------
    block_path : str
        the path to the block

    Returns
    ------

    trials : pandas dataframe

    Columns
    ------
    '''
    with h5py.File(core.get_kwik(block_path), 'r') as f:
        with f['event_types']['DigMark']['codes'].astype('str'):
            stims = pd.DataFrame({ 'time_samples' : f['event_types']['DigMark']['time_samples'],
                                   'code' : f['event_types']['DigMark']['codes'][:] })
        #assumes one start and one end for each trial
        stims.loc[stims.code == '<', 'stim_end_time_stamp'] = stims[stims['code'] == '>']['time_samples'].values 
        stims = stims[stims['code'] == '<']
        stims['stim_name'] = f['event_types']['Stimulus']['text'][1::2]
    stims['stim_presentation'] = stims[stims['code'] == '<']['stim_name'].map(_EventCounter().count)
    stims.reset_index(drop=True, inplace=True)
    del stims['code']
    return stims

def timestamp2time(df, sample_rate, time_stamp_label, time_label):
    df[time_stamp_label] = df[time_stamp_label].values / sample_rate
    df.rename(columns={time_stamp_label: time_label}, inplace=True)


def raster_by_unit(spikes, cluster, sample_rate, stim_length, window_size=1):
    sns.set_context("notebook", font_scale=1.5, rc={'lines.markeredgewidth': .1, 'patch.linewidth':1})
    sns.set_style("white")
    num_repeats = np.max(spikes['stim_presentation'].values)
    num_stims = len(np.unique(spikes["stim_name"]))
    g = sns.FacetGrid(spikes[spikes['cluster']==cluster], 
        col="stim_name", col_wrap=int(np.sqrt(num_stims)), 
        xlim=(-window_size, stim_length+window_size), ylim=(0,num_repeats));
    g.map(plt.scatter, "stim_aligned_time", "stim_presentation", marker='|')
    for ax in g.axes.flat:
        ax.plot((0, 0), (0, num_repeats), c=".2", alpha=.5)
        ax.plot((stim_length, stim_length), (0, num_repeats), c=".2", alpha=.5)
    g = g.set_titles("cluster %d, stim: {col_name}" % (cluster))
    plt.savefig('test.png')


from collections import Counter
class _EventCounter(Counter):
    def count(self, key):
        self[key] += 1
        return self[key] - 1

class _EventAligner(object):
    # ['time_stamp', 'stim_end_time_stamp', 'stim_name', 'stim_presentation']
    # TODO: make robust to multiple recordings
    # TODO: duplicate spikes that are <2 sec from 2 stimuli
    # TODO: generalize for any event type
    def __init__(self, stims, stim_index=0):
        self.stim_index = stim_index
        self.stims = stims
        self.prev_stim = None
        self.cur_stim = self.stims.loc[self.stim_index].values
        self.next_stim = self.stims.loc[self.stim_index+1].values
    def stim_checker(self, time_stamp):
        if time_stamp < self.cur_stim[0]:
            if self.stim_index == 0 or self.cur_stim[0] - time_stamp < time_stamp - self.prev_stim[1]:
                return self.cur_stim[[2, 3, 0]]
            else:
                return self.prev_stim[[2, 3, 0]]
        elif time_stamp < self.cur_stim[1]:
            return self.cur_stim[[2, 3, 0]]
        else:
            if self.stim_index + 1 < len(self.stims):
                #print time_stamp, self.stim_index
                self.stim_index += 1
                self.prev_stim = self.cur_stim
                self.cur_stim = self.next_stim
                if self.stim_index + 1 < len(self.stims):
                    self.next_stim = self.stims.loc[self.stim_index+1].values
                else:
                    self.next_stim = None
                return self.stim_checker(time_stamp)
            else:
                return self.cur_stim[[2, 3, 0]]