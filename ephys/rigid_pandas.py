import h5py
import pandas as pd
import numpy as np
import seaborn as sns

from collections import Counter
class StimCounter(Counter):
    def count(self, key):
        self[key] += 1
        return self[key] - 1

class Stims(object):
    # ['code', 'time_stamp', 'stim_end_time_stamp', 'stim_name', 'stim_presentation']
    # TODO: duplicate spikes that are <2 sec from 2 stimuli
    def __init__(self, stims, stim_index=0):
        self.stim_index = stim_index
        self.stims = stims
        self.prev_stim = None
        self.cur_stim = self.stims.loc[self.stim_index].values
        self.next_stim = self.stims.loc[self.stim_index+1].values
    def stim_checker(self, time_stamp):
        if time_stamp < self.cur_stim[1]:
            if self.stim_index == 0 or self.cur_stim[1] - time_stamp < time_stamp - self.prev_stim[2]:
                return self.cur_stim[[3, 4, 1]]
            else:
                return self.prev_stim[[3, 4, 1]]
        elif time_stamp < self.cur_stim[2]:
            return self.cur_stim[[3, 4, 1]]
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
                return self.cur_stim[[3, 4, 1]]

def load_kwik(kwikfile):
    with h5py.File(kwikfile, 'r') as f:
        sample_rate = None
        for recording in f['recordings']:
            assert sample_rate is None or sample_rate == f['recordings'][recording].attrs['sample_rate']
            sample_rate = f['recordings'][recording].attrs['sample_rate']
        spikes = pd.DataFrame({ 'time_stamp' : f['channel_groups']['0']['spikes']['time_samples'],
                                'cluster' : f['channel_groups']['0']['spikes']['clusters']['main'] })
        cluster_groups = {}
        for cluster in f['channel_groups']['0']['clusters']['main'].keys():
            cluster_groups[int(cluster)] = f['channel_groups']['0']['clusters']['main'][cluster].attrs['cluster_group']
        spikes['cluster_group'] = spikes['cluster'].map(cluster_groups)
        
        with f['event_types']['DigMark']['codes'].astype('str'):
            stims = pd.DataFrame({ 'time_stamp' : f['event_types']['DigMark']['time_samples'],
                                   'code' : f['event_types']['DigMark']['codes'][:] })
        stims.loc[stims.code == '<', 'stim_end_time_stamp'] = stims[stims['code'] == '>']['time_stamp'].values
        stims = stims[stims['code'] == '<']
        stims['stim_name'] = f['event_types']['Stimulus']['text'][1::2]
    stims['stim_presentation'] = stims[stims['code'] == '<']['stim_name'].map(StimCounter().count)
    stims.reset_index(drop=True, inplace=True)
    spikes['stim_name'], spikes['stim_presentation'], spikes['stim_time_stamp'] = zip(*spikes["time_stamp"].map(Stims(stims).stim_checker))
    spikes['stim_aligned_time_stamp'] = spikes['time_stamp'].values.astype('int') - spikes['stim_time_stamp'].values
    spikes['stim_aligned_time'] = spikes['stim_aligned_time_stamp'].values / sample_rate
    return spikes, stims, sample_rate

def raster_by_unit(spikes, cluster, sample_rate):
    sns.set_context("notebook", font_scale=1.5, rc={'lines.markeredgewidth': .1, 'patch.linewidth':1})
    sns.set_style("white")
    g = sns.FacetGrid(spikes[spikes['cluster']==cluster], col="stim_name", col_wrap=3, xlim=(-.5, 1), ylim=(0,num_repeats));
    g.map(plt.scatter, "stim_aligned_time", "stim_presentation", marker='|')
    for ax in g.axes.flat:
        ax.plot((0, 0), (0, num_repeats), c=".2", alpha=.5)
        ax.plot((stim_length, stim_length), (0, num_repeats), c=".2", alpha=.5)
    g = g.set_titles("cluster %d, stim: {col_name}" % (cluster))