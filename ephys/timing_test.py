import core, spikes, events, spiketrains
import os
from timeit import default_timer as timer

small_block = '/mnt/cube/ephys-example-data/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1'
large_block = '/mnt/cube/ephys-example-data/long_acute_rec'
behave_block = '/mnt/cube/ephys-example-data/Pen01_Lft_AP2500_ML1350__Site10_Z2026__B997_cat_P01_S10_Epc10'

print core.get_kwik(behave_block)

for block in [small_block, large_block, behave_block]:
#for block in [small_block]:
	start = timer()
	try:
		print "getting spikes"
		spikes = core.get_spikes(block)
		print "getting clusters"
		clusters = core.get_clusters(block)
		print "getting trials"
		trials = events.get_trials(block)
		fs = core.get_fs(block)
		print "getting spiketrains"
		for idx, cluster, quality in clusters.itertuples():
			for idx2, recording, time_sample, stimulus, stimulus_end, response, response_time, consequence, correct in trials.itertuples():
				spiketrain = spiketrains.get_spiketrain(recording, time_sample, cluster, spikes, (-1.0, (stimulus_end-time_sample)/fs +1.0), fs)
	finally:
		end = timer()
		print "%s took %d seconds or %d minutes" % (os.path.split(block)[-1], end-start, (end-start)/60.0)

