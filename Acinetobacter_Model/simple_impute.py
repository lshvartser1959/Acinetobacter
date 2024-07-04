import copy, math, os, pickle, time, numpy as np
#import ray
import pandas as pd
#
#
# ray.shutdown()
# ray.init(num_cpus=16)
ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']


def simple_imputer(df,train_subj=None,use_global_means=False):
    idx = pd.IndexSlice
    df = df.copy()
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    # icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    if use_global_means :
        with open('/mnt/data2/global_means', mode='rb') as f:
            global_means = pickle.load(f)
    else:
        if train_subj is None:
            global_means = df_out.loc[idx[:, :], idx[:, 'mean']].mean(axis=0)
        else:
            global_means = df_out.loc[idx[ train_subj,:], idx[:, 'mean']].mean(axis=0)


        with open('/mnt/data2/global_means', mode='wb') as f:
            pickle.dump(global_means, f, pickle.HIGHEST_PROTOCOL)


    df_out.loc[:, idx[:, 'mean']] = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).fillna(method='ffill')
    print('finished new impute')
    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out

if __name__ == '__main__':
   pass

