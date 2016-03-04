# functions to process stim & trial info from digmarks

def find_stim_end(digmark_row):
    return digmark_row['codes'] in '>#'
    
class FindEnd():
    def __init__(self):
        self.keep = True
    def check(self,code):
        if code in '()FfTt]N>#':
            self.keep = False
        return self.keep
    
def get_stim_start(stim_end_row,digmarks):
    rec,ts = stim_end_row['recording'],stim_end_row['time_samples']
    mask = (
        (digmarks['recording']==rec)
        & (digmarks['time_samples'] < ts)
        & ~digmarks['codes'].str.contains('[RCL]')
        )
    this_trial_mask = (
        digmarks[mask].iloc[::-1]['codes'].apply(FindEnd().check).iloc[::-1]
        & digmarks[mask]['codes'].str.contains('<')
    )
    this_trial = digmarks[mask][this_trial_mask]
    return this_trial.iloc[0]

def is_not_floatable(string):
    ''' returns True if string can be converted to float else False
    '''
    try: 
        float(string)
        return False
    except ValueError:
        return True


def get_stim_info(trial_row,stimulus,fs):
    rec,samps = trial_row['recording'], trial_row['time_samples']
    stim_mask = (
        (stimulus['recording']==rec)
        & (stimulus['time_samples']>(samps-0.1*fs))
        & (stimulus['time_samples']<(samps+fs))
        & ~stimulus['text'].str.contains('date')
        & stimulus['text'].apply(is_not_floatable)
        )
    
    if stim_mask.sum()>0:
        return stimulus[stim_mask].iloc[0]
    else:
        return dict(codes=np.nan,time_samples=np.nan,recording=np.nan,text=np.nan)


def get_stim_end(trial_row,digmarks,fs,window=60.0):
    rec,samps = trial_row['recording'], trial_row['time_samples']
    resp_mask = (
        (digmarks['recording']==rec)
        & (digmarks['time_samples']>samps)
        & (digmarks['time_samples']<(samps+fs*window))
        & digmarks['codes'].str.contains('[>#]')
        )
    assert digmarks[resp_mask].shape[0]>0
    return digmarks[resp_mask].iloc[0]

def get_response(trial_row,digmarks,fs,window=5.0):
    rec,samps = trial_row['recording'], trial_row['time_samples']
    try:
        stim_dur = trial_row['stimulus_end'] - trial_row['time_samples']
    except KeyError:
        stim_dur = get_stim_end(rec,samps,fs)['time_samples']-samps
    resp_mask = (
        (digmarks['recording']==rec)
        & (digmarks['time_samples']>(samps+stim_dur))
        & (digmarks['time_samples']<(samps+stim_dur+fs*window))
        & digmarks['codes'].str.contains('[RLN]')
        )
    if digmarks[resp_mask].shape[0]>0:
        return digmarks[resp_mask].iloc[0]
    else:
        return dict(codes=np.nan,time_samples=np.nan,recording=np.nan)

def get_consequence(trial_row,digmarks,fs,window=2.0):
    rec,samps = trial_row['recording'], trial_row['time_samples']
    rt = trial_row['response_time']
    bds = rt, rt+fs*window
    resp_mask = (
        (digmarks['recording']==rec)
        & (digmarks['time_samples']>bds[0])
        & (digmarks['time_samples']<bds[1])
        & digmarks['codes'].str.contains('[FfTt]')
        )
    if digmarks[resp_mask].shape[0]>0:
        return digmarks[resp_mask].iloc[0]
    else:
        return dict(codes=np.nan,time_samples=np.nan,recording=np.nan)

def is_correct(consequence):
    return consequence if consequence is np.nan else consequence in 'Ff'

def get_trials(block):
    digmarks = read_events(block,'DigMark')
    stimulus = read_events(block,'Stimulus')
    fs = get_fs(block)
    
    stim_end_mask = digmarks.apply(find_stim_end,axis=1)
    trials = digmarks[stim_end_mask].apply(lambda row: get_stim_start(row,digmarks),axis=1)[:]
    trials.reset_index(inplace=True)
    del trials['index']
    del trials['codes']
    trials['stimulus'] = trials.apply(lambda row: get_stim_info(row,stimulus,fs)['text'],axis=1)
    trials['stimulus_end'] = trials.apply(lambda row: get_stim_end(row,digmarks,fs)['time_samples'],axis=1)
    trials['response'] = trials.apply(lambda row: get_response(row,digmarks,fs)['codes'],axis=1)
    trials['response_time'] = trials.apply(lambda row: get_response(row,digmarks,fs)['time_samples'],axis=1)
    trials['consequence'] = trials.apply(lambda row: get_consequence(row,digmarks,fs)['codes'],axis=1)
    trials['correct'] = trials['consequence'].apply(is_correct)
    return trials