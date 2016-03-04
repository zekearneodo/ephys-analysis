import rigid_pandas
from timeit import default_timer as timer

kwikfile = '/home/mthielk/Data/B979/acute/analysis/klusta/Pen04_Lft_AP2555_ML500__Site02_Z2688__st979_cat_P04_S02_1/st979_cat_P04_S02_1.kwik'
kristas_kwikfile = '/mnt/cube/ephys-example-data/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/B1056_cat_P01_S03_1.kwik'

start = timer()
spikes, stims, sample_rate = rigid_pandas.load_kwik(kristas_kwikfile)
end = timer()
print end-start
#print spikes

spikes['morph_dim'] = spikes[~spikes.stim_name.str.contains('rec')].stim_name.str[0:2]
spikes['morph_pos'] = spikes[~spikes.stim_name.str.contains('rec')].stim_name.str[2:].astype(int)
good_spikes = spikes[spikes['cluster_group'] == 2]
#rec_stims = good_spikes[good_spikes.stim_name.str.contains('rec')]

