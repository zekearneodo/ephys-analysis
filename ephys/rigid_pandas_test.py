import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rigid_pandas
import core
from timeit import default_timer as timer

kwikfile = '/home/mthielk/Data/B979/acute/analysis/klusta/Pen04_Lft_AP2555_ML500__Site02_Z2688__st979_cat_P04_S02_1'
kristas_kwikfile = '/mnt/cube/ephys-example-data/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1'

block_path = kristas_kwikfile

start = timer()
spikes, stims = rigid_pandas.kwik2rigid_pandas(block_path)
end = timer()
print end-start
print spikes.head()
print stims.head()

#spikes['morph_dim'] = spikes[~spikes.stim_name.str.contains('rec')].stim_name.str[0:2]
#spikes['morph_pos'] = spikes[~spikes.stim_name.str.contains('rec')].stim_name.str[2:].astype(int)


#rec_stims = good_spikes[good_spikes.stim_name.str.contains('rec')]

clusters = core.get_clusters(block_path)
fs = core.get_fs(block_path)
stim_length = (stims.stim_end_time_stamp.head(1).values[0] - stims.time_samples.head(1).values[0]) / fs
print clusters.head()
cluster = clusters[clusters['quality']=="Good"]['cluster'].iloc[0]
print stim_length
rigid_pandas.raster_by_unit(spikes, cluster, fs, stim_length)
