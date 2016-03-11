import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import core

def kwik2rigid_pandas(block_path):
    spikes = core.load_spikes(block_path)
    stims = load_acute_stims(block_path)
    columns2copy = ['stim_name', 'stim_presentation', 'stim_start', 'stim_end']
    spike_stim_info_df = pd.DataFrame(data=list(spikes["time_samples"].map(_EventAligner(stims, output_labels=columns2copy).event_checker)), columns=columns2copy, index=spikes.index)
    spikes = spikes.join(spike_stim_info_df)
    #spikes['stim_name'], spikes['stim_presentation'], spikes['stim_time_stamp'] = zip(*spikes["time_samples"].map(_EventAligner(stims).stim_checker))
    spikes['stim_aligned_time'] = spikes['time_samples'].values.astype('int') - spikes['stim_start'].values
    spikes['stim_duration'] = spikes['stim_end'] - spikes['stim_start']
    fs = core.get_fs(block_path)
    timestamp2time(spikes, fs, 'stim_duration')
    del spikes['time_samples']
    del spikes['stim_start']
    del spikes['stim_end']
    timestamp2time(spikes, fs, 'stim_aligned_time')
    return spikes, stims

def load_acute_stims(block_path):
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

    stim_start : int
        Time in samples of the start of a stimulus (trial)
    stim_name : str
        Name of the stimulus
    stim_end : int
        Time in samples of the end of the stimulus
    stim_presentation : int
        presentation number, starting at 0, going to # of presentations-1
    '''
    with h5py.File(core.find_kwik(block_path), 'r') as f:
        with f['event_types']['DigMark']['codes'].astype('str'):
            stims = pd.DataFrame({ 'stim_start' : f['event_types']['DigMark']['time_samples'],
                                   'code' : f['event_types']['DigMark']['codes'][:] })
        #assumes one start and one end for each trial
        stims.loc[stims.code == '<', 'stim_end'] = stims[stims['code'] == '>']['stim_start'].values 
        stims = stims[stims['code'] == '<']
        stims['stim_name'] = f['event_types']['Stimulus']['text'][1::2]
    stims['stim_presentation'] = stims[stims['code'] == '<']['stim_name'].map(_EventCounter().count)
    stims.reset_index(drop=True, inplace=True)
    del stims['code']
    return stims

def timestamp2time(df, sample_rate, time_stamp_label, time_label=None, inplace=True):
    if inplace:
        df[time_stamp_label] = df[time_stamp_label].values / sample_rate
        if time_label:
            df.rename(columns={time_stamp_label: time_label}, inplace=True)
    else:
        assert time_label, 'must provide time_label if not inplace'
        df[time_label] = df[time_stamp_label].values / sample_rate



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
    def __init__(self, events, output_labels, start_label='stim_start', end_label='stim_end', event_index=0):
        self.event_index = event_index
        self.events = events

        event_columns = list(events.keys().get_values())
        self.output_indices = [event_columns.index(lbl) for lbl in output_labels]
        self.start_index = event_columns.index(start_label)
        self.end_index = event_columns.index(end_label)

        self.prev_event = None
        self.cur_event = self.events.loc[self.event_index].values
        self.next_event = self.events.loc[self.event_index+1].values
    def event_checker(self, time_stamp):
        if time_stamp < self.cur_event[self.start_index]:
            if self.event_index == 0 or self.cur_event[self.start_index] - time_stamp < time_stamp - self.prev_event[self.end_index]:
                return self.cur_event[self.output_indices]
            else:
                return self.prev_event[self.output_indices]
        elif time_stamp < self.cur_event[self.end_index]:
            return self.cur_event[self.output_indices]
        else:
            if self.event_index + 1 < len(self.events):
                #print time_stamp, self.event_index
                self.event_index += 1
                self.prev_event = self.cur_event
                self.cur_event = self.next_event
                if self.event_index + 1 < len(self.events):
                    self.next_event = self.events.loc[self.event_index+1].values
                else:
                    self.next_event = None
                return self.event_checker(time_stamp)
            else:
                return self.cur_event[self.output_indices]