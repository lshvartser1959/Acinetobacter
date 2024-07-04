from common import*
from functions import*
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from simple_impute import simple_imputer
import scipy.stats as ss #Shua
from xgboost import XGBClassifier
import sklearn
import optuna
import os
from AUCROC import Gntbok1
import pickle

import timeit

def mksureDir(outdir):
    try:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            print(os.getcwd())
            print(os.path.exists(outdir))
        else:
            print(os.getcwd())
            print(os.path.exists(outdir))
    except:
        pass


def interval_labelling(outcomes):
    IDS = list(outcomes.loc[:].groupby(['subject_id', 'hadm_id', 'icustay_id']).mean().index)
    df_dict = dict.fromkeys(IDS, None)
    nIDS = len(IDS)
    columns = [str(col) for col in outcomes.columns]
    cols = INTERVENTION + ['clean bacteria spec']

    ii = 0
    for i in IDS:
        print(str(ii)+ " from "+ str(nIDS))
        df = outcomes.loc[i]
        column_ids = [df.columns.get_loc(c) for c in INTERVENTION + ['clean bacteria spec'] if c in df]

        start_to_numpy = timeit.default_timer()
        np_df = df.to_numpy()
        stop_to_numpy = timeit.default_timer()

        print('Time to_numpy: ', stop_to_numpy - start_to_numpy)

        start_to_fill = timeit.default_timer()
        for j in column_ids:
            starts = np.where(np_df[j] == 1)[0] - PREDICTION_WINDOW + 1
            ends = np.where(np_df[j] == 1)[0]
            for k in range(len(starts)):
                s = max(starts[k], 0)
                e = max(ends[k], 0)
                for kk in range(s, e):
                    np_df[j][kk] = 1
        stop_to_fill = timeit.default_timer()
        print('Time to_fill: ', stop_to_fill - start_to_fill)

        start_to_add = timeit.default_timer()
        df_dict[i] = np_df

        id_df =  df.index
        ids = [[i[0], i[1], i[2], x] for x in id_df]

        if ii==0:
#                np_A = np_s
            ids_A = ids
        else:
#                np_A = np.concatenate((np_A, np_s))
            ids_A = ids_A +ids
        stop_to_add = timeit.default_timer()
        print('Time to_add: ', stop_to_add - start_to_add)

        ii += 1
    df_list = list(df_dict.values())
    start_concat = timeit.default_timer()

    out = np.concatenate(df_list)

    stop_concat = timeit.default_timer()

    print('Time: ', stop_concat - start_concat)

    idds = [tuple(x) for x in ids_A]
    multi_index = pd.MultiIndex.from_tuples(idds)
    multi_index = multi_index.set_names(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])

    start_X_shifted = timeit.default_timer()
    out = pd.DataFrame(out, index=multi_index, columns=columns)

    stop_X_shifted = timeit.default_timer()

    print('Time: ', stop_X_shifted - start_X_shifted)

    return out


import h5py

def load_data(DATAFILE):
    print('\nOK!\nReading static ')
    print(os.getcwd())
    print(os.getcwd())
    # f1 = h5py.File(DATAFILE, 'r+')
    #
    # KEYS = [key for key in f1.keys()]

    # load inputevents
    print('\nOK!\nReading c_df... ')
    c_df = pd.read_hdf(DATAFILE, 'c_df')
    print('\nOK!\nReading inputevents... ')
    inputevents = pd.read_hdf(DATAFILE, 'inputevents')

    # Load static
    static = pd.read_hdf(DATAFILE, 'patients')

    # Load vitals labs
    print('\nOK!\nReading vitals_labs... ')
    X = pd.read_hdf(DATAFILE, 'vitals_labs')
    print('\nOK! ->>>>>' + str(X.shape[1]) + ' vitals_lab found...\n ')

    # Load interventions
    print('\nOK!\nReading interventions... ')
    Y = pd.read_hdf(DATAFILE, 'interventions')
    # Y = Y[['vent']]   WHy is this here? It gets rid of all the data for add2static -- Shua
    Y = Y[INTERVENTION]
    ### JJ
    outcomes = pd.read_hdf(DATAFILE, 'interventions')

    # One of 4 Y - Irrelevant
    FOUR_STEP_INTERVAL=False
    if FOUR_STEP_INTERVAL:
        TO_WRITE = True

        outcomes = interval_labelling(outcomes)
        if TO_WRITE:
            a_file = open('/mnt/data2/' + "outcomes.pkl", "wb")
            pickle.dump(outcomes, a_file)
            a_file.close()
        else:
            a_file = open('/mnt/data2/' + "outcomes.pkl", "rb")
            outcomes = pickle.load(a_file)
            a_file.close()

    # rows_to_keep : at list one of bacterias - positives,  'clean bacteria spec' - negatives
    rows_to_keep = ((outcomes[INTERVENTION].any(axis=1)) | (outcomes['clean bacteria spec']))

    # X + inputevents
    X = pd.concat([X, inputevents], axis=1)

    ###
    print('Shape of X : ', X.shape)
    print('Shape of Y : ', Y.shape)
    print('Shape of static : ', static.shape)

    return X,Y, static, rows_to_keep, outcomes ### JJ
def filter_data(X, Y, static, rows_to_keep,outcomes):
    r2k = rows_to_keep[rows_to_keep == 1].reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).size()
    r2k.name = 'r2k'
    static = static.join(r2k, how='inner').drop('r2k', axis=1)
    Y = Y.join(r2k, how='inner').drop('r2k', axis=1)
    rows_to_keep = pd.DataFrame(rows_to_keep).join(r2k, how='inner').drop('r2k', axis=1)
    outcomes = outcomes.join(r2k, how='inner').drop('r2k', axis=1)

    r2k2 = pd.DataFrame(r2k)
    r2k2.columns = pd.MultiIndex.from_product([r2k2.columns, ['']])
    X2 = X.join(r2k2, how='inner').iloc[:, :-1]
    X2.index = X2.index.remove_unused_levels()  # X2 replaces X. Keeping it as seperate variable for debugging

    return X2, outcomes, Y, static, rows_to_keep
    ##X2
def add2static(Y, static):
    # make some temporary operation....... to see counts of vent hours per patient....
    VY = Y.reset_index()
    VY = VY.set_index('subject_id')
    VY = VY.drop('icustay_id', axis=1)
    VY = VY.drop('hadm_id', axis=1)
    # yag = VY[['hours_in', 'acinetobacter']].groupby(['subject_id']).agg({'hours_in': ['count'], 'vent': ['sum']})
    yag = VY.groupby(['subject_id']).sum()[INTERVENTION]

    # if yag > 0 so patient was ventilated
    yag = yag > 0

    # Set true or false for ventilation... in static
    # static['acinetobacter'] = yag.acinetobacter.values
    static = static.join(yag)  # Shua-- This is just temporary, while only taking some of the patients and not all. Otherwise, an error of unequal merging lengths is

    # GC...
    del VY, yag

    lengths = np.array(static.reset_index().max_hours + 1).reshape(-1)

    return static, lengths

def create_long_shifted_vectors(X_merge, rows_to_keep,gap_size, name):
    if name=='std':
        X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).std()
    if name=='min':
        X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).min()
    if name=='max':
        X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).max()
    if name=='var':
        X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).var()
    if name=='skew':
        X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).skew()
    if name=='kurt':
        X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).kurt()
    if name=='cov':
        X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).cov()
    if name=='corr':
        X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).corr()


    ###================= LEONID 27.06.2022 =========================================================#
    # X_shifted2 = X_shifted.drop(['subject_id', 'hadm_id', 'icustay_id'], axis=1).reset_index().set_index(
    #     ['subject_id', 'hadm_id', 'icustay_id']).shift(gap_size - 1)
    X_shifted.index.name = None
    X_shifted2 = X_shifted.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id']).shift(gap_size - 1)
    X_shifted2=X_shifted2.drop('level_3',axis=1)
    ###================= LEONID 27.06.2022 =========================================================#

    X_shifted2.reset_index(inplace=True)
    X_shifted2['hours_in'] = X_merge.reset_index()['hours_in']
    X_shifted2.set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'], inplace=True)
    rows_to_keep.columns = ['rows_to_keep']
    X_shifted3 = X_shifted2.join(rows_to_keep)
    x = X_shifted3[X_shifted3.rows_to_keep > 0].drop('rows_to_keep', axis=1)

    return x, rows_to_keep

def window_stack(a, stepsize=1, width=SLICE_SIZE):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

def create_long_window(X_merge, Y, rows_to_keep,gap_size, path):
    # Inputs X_merge and Y from when X_merge one set of features against one outcome of Y according to the time
    # Outputs are X with following properties:
    #       a. consisted of SLICE_SIZE time repeated features on interval [t-SLICE_SIZE, t]
    #       b. rows in X shifted SLICE_SIZE+gap_size forward: data against every element in Y is SLICE_SIZE+gap_size before
    #       c. remaied only rows that are in rows_to_keep
    #             Y remaied only rows that are in rows_to_keep

    # Create list of ['subject_id', 'hadm_id', 'icustay_id'] for X_merge
    IDS = list(X_merge.loc[:].groupby(['subject_id', 'hadm_id', 'icustay_id']).mean().index)
    # Prepare dictionary to get data by id
    df_dict = dict.fromkeys(IDS, None)
    nIDS = len(IDS)

    # CALC_TENSOR = False when df_dict are already done , only to read
    CALC_TENSOR = True
    # CALC_SHIFTED = False when X_shifted are already done, only to read
    CALC_SHIFTED = True
    if CALC_SHIFTED:
        # features except age and sex are from 0 to -5 hours from their time
        columns5 = [str(col) for col in X_merge.columns if
                    col not in ["gender_F", "gender_M", "age_1", "age_2", "age_3", "age_4"]]
        columns4 = [str(col) + '_1' for col in columns5]
        columns3 = [str(col) + '_2' for col in columns5]
        columns2 = [str(col) + '_3' for col in columns5]
        columns1 = [str(col) + '_4' for col in columns5]
        columns0 = [str(col) + '_5' for col in columns5]
        columns = columns0 + columns1 + columns2 + columns3 + columns4 + columns5 + ["gender_F", "gender_M", "age_1",
                                                                                     "age_2", "age_3", "age_4"]
        if CALC_TENSOR:

            ii = 0
            for i in IDS:
                print(str(ii)+ " from "+ str(nIDS))
                df = X_merge[(X_merge.index.get_level_values('subject_id')==i[0])*(X_merge.index.get_level_values('hadm_id')==i[1])*(X_merge.index.get_level_values('icustay_id')==i[2])]
                df_static = df[["gender_F", "gender_M", "age_1", "age_2", "age_3", "age_4"]]
                # Remain only statics with time not less than SLICE_SIZE
                filtered_static = df_static[df_static.index.get_level_values('hours_in')>=SLICE_SIZE-1]
                df = df.drop(["gender_F", "gender_M", "age_1", "age_2", "age_3", "age_4"],axis=1)
                # dynamics and statics to numpy array
                np_df = df.to_numpy()
                np_st = filtered_static.to_numpy()
                id_df =  df.index[SLICE_SIZE-1:np_df.shape[0]]
                # Shift ids
                ids = [[i[0], i[1], i[2], x] for x in id_df.get_level_values('hours_in')]
                # Make window step=1 , width=SLICE_SIZE
                np_s = window_stack(np_df, stepsize=1, width=SLICE_SIZE)
                np_s = np.hstack((np_s, np_st))
                # fill ids and df_dict
                if ii==0:
                    ids_A = ids
                else:
                    ids_A = ids_A +ids


                df_dict[i] = np_s

                ii += 1

            # store ids and df_dict
            a_file = open('/mnt/data2/' + "df_dict.pkl", "wb")
            pickle.dump(df_dict, a_file)
            a_file.close()
            b_file = open('/mnt/data2/' + "ids_A.pkl", "wb")
            pickle.dump(ids_A, b_file)
            b_file.close()


        else:
            # read ids and df_dict
            a_file = open('/mnt/data2/' + "df_dict.pkl", "rb")
            df_dict = pickle.load(a_file)
            a_file.close()
            b_file = open('/mnt/data2/' + "ids_A.pkl", "rb")
            ids_A = pickle.load(b_file)
            b_file.close()

        df_list = list(df_dict.values())
        start_concat = timeit.default_timer()

        # Create numpy array from data
        X_sh = np.concatenate(df_list)

        stop_concat = timeit.default_timer()

        print('Time: ', stop_concat - start_concat)

        # Create multiple index from indices
        idds = [tuple(x) for x in ids_A]
        multi_index = pd.MultiIndex.from_tuples(idds)
        multi_index = multi_index.set_names(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])

        # Create data frame from numpy array, multiple indices and column names
        start_X_shifted = timeit.default_timer()
        X_shifted = pd.DataFrame(X_sh, index=multi_index, columns=columns)

        stop_X_shifted = timeit.default_timer()

        print('Time: ', stop_X_shifted - start_X_shifted)

        del X_sh, multi_index, X_merge

        # Save X_shifted
        print('X_shifted started')
#        X_shifted.to_hdf(path + 'X_shifted_'+str(gap_size)+'.hf5', 'tensor')
        start_X_shifted_read = timeit.default_timer()
        a_file = open('/mnt/data2/' + 'X_shifted_'+str(gap_size)+'.pkl', 'wb')
        pickle.dump(X_shifted, a_file)
        a_file.close()
        stop_X_shifted_read = timeit.default_timer()

        print('Time: ', stop_X_shifted_read - start_X_shifted_read)

        print('X_shifted finished')
    else:
        # Read X_shifted

        print('X_shifted read started')
        start_X_shifted_read = timeit.default_timer()

        a_file = open('/mnt/data2/' + 'X_shifted_'+str(gap_size)+'.pkl', 'rb')
        X_shifted = pickle.load(a_file)
        a_file.close()

#        X_shifted = pd.read_hdf(path + 'X_shifted_'+str(gap_size)+'.hf5', 'tensor')
        stop_X_shifted_read = timeit.default_timer()

        print('Time: ', stop_X_shifted_read - start_X_shifted_read)
        print('X_shifted read finished')
    ###================= Shift X_shifted in time by  (gap_size - 1) =========================================================#
    print('X_sh')
    X_sh = X_shifted.reset_index()['hours_in']
    print('X_sh1')
    X_sh1 = X_shifted.index
    X_shifted.index.name = None
    print('X_shifted2')
    X_shifted2 = X_shifted.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id']).shift(gap_size - 1)
    del X_shifted
    print('X_shifted2.drop')
    X_shifted2=X_shifted2.drop('hours_in',axis=1)
    print('X_shifted2.drop stop')

    ###================= Remain in X only rows_to_keep =========================================================#

    rows_to_keep_cut = rows_to_keep[rows_to_keep.index.isin(X_sh1)]
    X_shifted2.reset_index(inplace=True)
    X_shifted2['hours_in'] = X_sh
    X_shifted2.set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'], inplace=True)

    rows_to_keep_cut.columns = ['rows_to_keep']
    X_shifted3 = X_shifted2.join(rows_to_keep_cut)
    x = X_shifted3[X_shifted3.rows_to_keep > 0].drop('rows_to_keep', axis=1)

### ==================== Remain in Y only rows_to_keep    ==================================####
    Y2 = pd.DataFrame(Y.max(axis=1))
    Y2 = Y2[Y2.index.isin(X_sh1)]
    Y3 = Y2.join(rows_to_keep_cut)
    y = Y3[Y3.rows_to_keep > 0].drop('rows_to_keep', axis=1)
    return x, y, rows_to_keep_cut

def create_rolling_window(X_merge, Y, rows_to_keep,gap_size):
    X_shifted = X_merge.reset_index().groupby(['subject_id', 'hadm_id', 'icustay_id']).rolling(SLICE_SIZE).mean()

    ###================= LEONID 27.06.2022 =========================================================#
    # X_shifted2 = X_shifted.drop(['subject_id', 'hadm_id', 'icustay_id'], axis=1).reset_index().set_index(
    #     ['subject_id', 'hadm_id', 'icustay_id']).shift(gap_size - 1)
    X_shifted.index.name = None
    X_shifted2 = X_shifted.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id']).shift(gap_size - 1)
    X_shifted2=X_shifted2.drop('level_3',axis=1)
    ###================= LEONID 27.06.2022 =========================================================#

    X_shifted2.reset_index(inplace=True)
    X_shifted2['hours_in'] = X_merge.reset_index()['hours_in']
    X_shifted2.set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'], inplace=True)
    rows_to_keep.columns = ['rows_to_keep']
    X_shifted3 = X_shifted2.join(rows_to_keep)
    x = X_shifted3[X_shifted3.rows_to_keep > 0].drop('rows_to_keep', axis=1)

### ====================incorrect 1 of 4    ==================================####
    Y2 = pd.DataFrame(Y.max(axis=1))
    Y3 = Y2.join(rows_to_keep)
    y = Y3[Y3.rows_to_keep > 0].drop('rows_to_keep', axis=1)
    return x, y, rows_to_keep
def train_test_val(static, random_seed):
    ## Train-Test Split, Stratified

    # %%
    print('\nOK!\n Splitting Data....','using seed:',random_seed)
    train_ids, rest_ids = train_test_split(static.reset_index(), test_size=TEST_SIZE,
                                           random_state=random_seed, stratify=static[INTERVENTION].any(axis=1))
    test_ids, val_ids = train_test_split(rest_ids, test_size=0.5,
                                                random_state=random_seed, stratify=rest_ids[INTERVENTION].any(axis=1))
    print(INTERVENTION, "' Balanced - > ", 'count: ', train_ids[train_ids[INTERVENTION] > 0].shape[0], 'rate: ',
          (train_ids[train_ids[INTERVENTION] > 0].shape[0] / train_ids.shape[0]))

    # %%

    print('\nOK!\n ')


    return train_ids, val_ids, test_ids


def imputation(data, train=None):
    print('\nOK!\n simple_imputer .............')
    X_clean = simple_imputer(data,train)
    X_clean = X_clean.fillna(0)

    return X_clean

# def normalization(data):
def normalization(data, train_ids=None):
    idx = pd.IndexSlice
    X_std = data.copy()
    # pd.read_csv()
    X_std.loc[:, idx[:, 'mean']] = X_std.loc[:, idx[:, 'mean']].apply(lambda x: minmax(x))
    # X_std.loc[:, idx[:, 'count']] = X_std.loc[:, idx[:, 'count']].apply (lambda x: minmax (x))
    # X_std.loc[:, idx[:, 'median']] = X_std.loc[:, idx[:, 'median']].apply (lambda x: minmax (x))
    # X_std.loc[:, idx[:, 'min']] = X_std.loc[:, idx[:, 'min']].apply (lambda x: minmax (x))
    # X_std.loc[:, idx[:, 'max']] = X_std.loc[:, idx[:, 'max']].apply (lambda x: minmax (x))
    X_std.loc[:, idx[:, 'time_since_measured']] = X_std.loc[:, idx[:, 'time_since_measured']].apply(
        lambda x: std_time_since_measurement(x))

    # %%

    XScols = X_std.columns.values

    X_std.columns = [' '.join(col).strip() for col in X_std.columns.values] #JJ

    print("length of X_std, ", X_std.shape)
    return X_std, XScols

def merge(data, static):
    # use gender, first_careunit, age and ethnicity for prediction
    static_to_keep = static[['gender', 'age']]#, 'ethnicity', 'first_careunit']]
    # static_to_keep.loc[:, 'intime'] = static_to_keep['intime'].astype('datetime64').apply(lambda x: x.hour)
    static_to_keep.loc[:, 'age'] = static_to_keep['age'].apply(categorize_age)
    #static_to_keep.loc[:, 'ethnicity'] = static_to_keep['ethnicity'].apply(categorize_ethnicity)
    static_to_keep = pd.get_dummies(static_to_keep, columns=['gender', 'age',])# 'ethnicity', 'first_careunit'])

    ## Create Feature Matrix

    # %%
    #print('\nOK!\n X_merge .............')
    # merge time series and static data
    X_merge = pd.merge(data.reset_index(), static_to_keep.reset_index(), on=['subject_id', 'icustay_id', 'hadm_id'])
    # add absolute time feature
    # abs_time = (X_merge['intime'] + X_merge['hours_in']) % 24
    # X_merge.insert(4, 'absolute_time', abs_time)
    # X_merge.drop('intime', axis=1, inplace=True)
    X_merge = X_merge.set_index(['subject_id', 'icustay_id', 'hadm_id', 'hours_in'])

    # time_series_col = 124  # X_merge.shape[1] - static_to_keep.shape[1] - 1 # ROY FIX - HARDCODED !! Ned to be calculated!
#     time_series_col: int = int(2 + X_merge.shape[1] - static_to_keep.shape[1])
    time_series_col = int(2 + X_merge.shape[1] - static_to_keep.shape[1])

    # tsc : int = int  (2+X_merge.shape[1] - static_to_keep.shape[1])

    print('time_series_col--------------> ', time_series_col);



    return X_merge, time_series_col, static_to_keep

class DictDist():
    def __init__(self, dict_of_rvs): self.dict_of_rvs = dict_of_rvs

    def rvs(self, n):
        a = {k: v.rvs(n) for k, v in self.dict_of_rvs.items()}
        out = []
        for i in range(n): out.append({k: vs[i] for k, vs in a.items()})
        return out



class Objective: #Shua-- Added this instead of the global variables being used. Needed a way to pass in the datasets to the Optuna objective function. Globals didn't allow us to comment out these functions when just doing the training
    def __init__(self, x, y, x_test, y_test):
        self.x = x
        self.y= y
        self.x_test = x_test
        self.y_test = y_test

    def __call__(self, trial):
        x_train = self.x
        y_train = self.y
        x_test = self.x_test
        y_test = self.y_test
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1)
        n_estimators = trial.suggest_int('n_estimators', 300,1000)
        max_depth = trial.suggest_int("max_depth", 3, 99, step=2)
        scale_pos_weight = int(trial.suggest_loguniform('scale_pos_weight', 1, 10000))
        min_child_weight = trial.suggest_int("min_child_weight", 2, 3)
        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                            min_child_weight = min_child_weight, gamma = gamma, n_jobs = 60, scale_pos_weight=scale_pos_weight)

        # if trial.should_prune():
        #     raise optuna.TrialPruned()

        clf.fit(x_train, y_train)
        y_pred = pd.DataFrame(clf.predict_proba(x_test))

        #TEMPORARY, ONLY FOR 100 PATIENS+++++++++++++++++++++++++
       # y_test.iloc[0] = 1
        #+++++++++++++++++++++++++++++++++++++++++++

        accuracy = sklearn.metrics.roc_auc_score(y_test, y_pred.iloc[:, 1])
        # print('len of feature_importances:', len(M.feature_importances_))
        # print('initial test score: ', accuracy)
        return accuracy

def training_testing(x_train, x_val, x_test, y_train, y_val, y_test, model, RANDOM, GAP_SIZE,
                     hyp_search = False, grid_search=False, do_adapt=0,n_trials=5):

    np.random.seed(RANDOM)
    # constant hyperparameters
    if hyp_search == False and grid_search == False:
        if model == 'no model':
            XGB_hyperparams_list = {'learning_rate': 0.0028976446685716, 'n_estimators': 466, 'max_depth': 5,
             'colsample_bytree': 0.6066635754661189, 'scale_pos_weight': 13, 'min_child_weight': 2,
             'gamma': 0.0003005922274502197}
            XGB_hyperparams_list = {'learning_rate': 0.08664382243964439, 'n_estimators': 525, 'max_depth': 5, 'scale_pos_weight': 13,
                   'min_child_weight': 2, 'gamma': 3.861843289374542e-05}
            XGB_hyperparams_list = {'learning_rate': 0.03684843956512154, 'n_estimators': 393, 'max_depth': 67,
             'colsample_bytree': 1, 'scale_pos_weight': 7.442337360635147, 'min_child_weight': 2,
             'gamma': 5.542749469938337e-08,'n_jobs' : 60}

        else:
            XGB_hyperparams_list = model.get_params()

        model_name, model, hyperparams_list = ('XGB', XGBClassifier, XGB_hyperparams_list)
        print('Initial test, with hardcoded hyperparameters')
        M = model(**hyperparams_list)
        print(XGB_hyperparams_list)
        M.fit(x_train, y_train)
        # y_pred = pd.DataFrame(model.predict_proba(x_test_concat))
        y_pred = pd.DataFrame(M.predict_proba(x_test))
        Y_test = pd.DataFrame(y_test)

        from sklearn.metrics import roc_auc_score
        print(roc_auc_score(Y_test, y_pred.iloc[:, 1]))
    # grid search for
    if grid_search:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        parameters = { 'learning_rate': [1e-3, 1e-2, 1e-1], #Hardcoded Optuna paramters on gap 6, random seed 1
                      'n_estimators': [300,400, 500, 600, 700, 800, 900, 1000],
                      'max_depth': [3,9,13,19,23,29,33,39,43,49,53,59,63,69,73,79,83,89,93, 99],
                      'min_child_weight': [2, 3],
                      'colsample_bytree': [0.6066635754661189, 0.7],
                      'scale_pos_weight': [1, 10, 100, 500, 1000, 3000, 5000, 7000, 10000],
                      'eval_metric': ['auc'],
                      'gamma': [1e-8, 1e-5, 1e-1, 1.0]
                      }

        # parameters = {'learning_rate': [0.042278005068887535], 'n_estimators': 463,
        #                                                 'max_depth': 5, 'scale_pos_weight': 6987.763179193735,
        #                                                 'min_child_weight': 2,
        #                                                 'gamma': 0.09329587166004492}

        #XGB_clf = GridSearchCV(XGBClassifier(), parameters, cv=3, verbose=1, n_jobs = -1)
        XGB_clf = RandomizedSearchCV(XGBClassifier(), parameters, cv=3, verbose=1, n_jobs = -1)
        XGB_clf.fit(np.nan_to_num(x_train), y_train)
        print(XGB_clf.best_params_)
        model_name, model, hyperparams_list = ('XGB', XGBClassifier, XGB_clf.best_params_)
        M = model(**hyperparams_list)
        print(hyperparams_list)
        M.fit(x_train, y_train)
        y_pred = pd.DataFrame(M.predict_proba(x_test))
        Y_test = pd.DataFrame(y_test)

#        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, confusion_matrix
        print('Train score:', roc_auc_score(pd.DataFrame(y_train), pd.DataFrame(M.predict_proba(x_train)).iloc[:, 1]))
        print('Test score:', roc_auc_score(Y_test, y_pred.iloc[:, 1]))

    # bayesian search
    if hyp_search:
        # Form objective with XGBoost
        objective = Objective(x_train, y_train, x_val, y_val)
        # Make the sampler behave in a deterministic way.
        sampler = optuna.samplers.TPESampler(seed=10)
        study = optuna.create_study(direction='maximize',sampler=sampler)
        # optuna running hyperparameter optimization
        study.optimize(objective, n_trials=n_trials,n_jobs= -1)
        trial = study.best_trial

        # Repeat train/test with optimal model
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        hyperparams_list = {key: value for key, value in trial.params.items()}
        M = XGBClassifier(**hyperparams_list)
        M.fit(x_train, y_train)
        y_pred = pd.DataFrame(M.predict_proba(x_test))
        Y_test = pd.DataFrame(y_test)

    from sklearn.metrics import roc_auc_score

    # test_score =  = AUC
    print('len of feature_importances:', len(M.feature_importances_))

    # TEMPORARY, ONLY FOR 100 PATIENS+++++++++++++++++++++++++
  #  Y_test.iloc[0] = 1
    # +++++++++++++++++++++++++++++++++++++++++++

    test_score = roc_auc_score(Y_test, y_pred.iloc[:, 1])
    print('Hyperparemeter search score:', roc_auc_score(Y_test, y_pred.iloc[:, 1]))
    return M, M.feature_importances_, Y_test, y_pred, test_score
    if do_adapt:
        # adapt TO ADD !!!
        a=0

    else:
        # M = model(**hyperparams_list)
        # M = model(**hyperparams_list)
        # print(XGB_hyperparams_list)
        # M.fit(x_train, y_train)
        # y_pred = pd.DataFrame(model.predict_proba(x_test_concat))
        y_pred = pd.DataFrame(M.predict_proba(x_test))
        Y_test = pd.DataFrame(y_test)
    from sklearn.metrics import roc_auc_score
    print('len of feature_importances:', len(M.feature_importances_))
    # print('initial test score: ', roc_auc_score(Y_test, y_pred.iloc[:, 1]))
    test_score =  roc_auc_score(Y_test, y_pred.iloc[:, 1])

    sns.set(font_scale=1.4)

    sns.heatmap(confusion_matrix(Y_test, np.round(y_pred.iloc[:, 1].values)), cmap="BuPu", annot=True, fmt="d",
                annot_kws={"size": 16})
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('True', fontsize=18)
    plt.show()
    return M, M.feature_importances_, Y_test, y_pred,test_score

def quality_characteristics(pred, test, GAP, SLICE_SIZE, PREDICTION_WINDOW):
    return Gntbok1(pred, test, GAP, SLICE_SIZE, PREDICTION_WINDOW)

def by_multiindex(x, ids):
    a = x.index.get_level_values('subject_id').isin(ids['subject_id'])
    b = x.index.get_level_values('hadm_id').isin(ids['hadm_id'])
    c = x.index.get_level_values('icustay_id').isin(ids['icustay_id'])
    return a*b*c

def tvt_split(x,y,train_ids, val_ids, test_ids):
    x_train = x[by_multiindex(x, train_ids)]
    x_val = x[by_multiindex(x, val_ids)]
    x_test = x[by_multiindex(x, test_ids)]
    y_train = y[by_multiindex(y, train_ids)]
    y_val = y[by_multiindex(y, val_ids)]
    y_test = y[by_multiindex(y, test_ids)]

    return x_train,x_val,x_test,y_train,y_val,y_test
