from common import*
import numpy as np
# %%
import pandas as pd
CHUNK_KEY = {'ONSET': 0, 'CONTROL': 1}

global minmaxlst
minmaxlst=[]

global minmaxlstIDX #Shua
minmaxlstIDX = 0

global stdlist
stdlist=[]

global stdlstIDX
stdlstIDX =0

# def std_time_since_measurement(x,loadedstdlist, readSTD = 0):
def std_time_since_measurement(x,loadedstdlist = None, readSTD = 0):

    global stdlist
    if not readSTD:
        global stdlstIDX
        idx = pd.IndexSlice
        x = np.where(x == 100, 0, x)
        means = x.mean()
        stds = x.std()
        x_std = (x - means) / stds
        stdlist.insert(stdlstIDX, [means,stds])
        stdlstIDX+=1;
    else:

        x_std =   (x - loadedstdlist[stdlstIDX][0]) / loadedstdlist[stdlstIDX][1]
        stdlstIDX += 1;

    return x_std


# def minmax(x,loadedminmaxlist= None, readSTD = 0):  # normalize
def minmax(x, loadedminmaxlist=None, readSTD=0):  # Shua

    global minmaxlst
    if not readSTD:
        global minmaxlstIDX
        mins = x.min()
        maxes = x.max()
        x_std = (x - mins) / (maxes - mins)
        minmaxlst.insert(minmaxlstIDX,[mins,maxes])
        minmaxlstIDX += 1;

    else:
        x_std = (x - loadedminmaxlist[minmaxlstIDX][0]) / (loadedminmaxlist[minmaxlstIDX][1] - loadedminmaxlist[minmaxlstIDX][0])
        minmaxlstIDX += 1;


    return x_std

## Categorization of Static Features

# %%

def categorize_age(age):
    if age > 10 and age <= 30:
        cat = 1
    elif age > 30 and age <= 50:
        cat = 2
    elif age > 50 and age <= 70:
        cat = 3
    else:
        cat = 4
    return cat


def categorize_ethnicity(ethnicity):
    if 'AMERICAN INDIAN' in ethnicity:
        ethnicity = 'AMERICAN INDIAN'
    elif 'ASIAN' in ethnicity:
        ethnicity = 'ASIAN'
    elif 'WHITE' in ethnicity:
        ethnicity = 'WHITE'
    elif 'HISPANIC' in ethnicity:
        ethnicity = 'HISPANIC/LATINO'
    elif 'BLACK' in ethnicity:
        ethnicity = 'BLACK'
    else:
        ethnicity = 'OTHER'
    return ethnicity

## Make Tensors

# %%

def create_x_matrix(x):
    zeros = np.zeros((MAX_LEN, x.shape[1] - 4))
    x = x.values
    x = x[:(MAX_LEN), 4:]
    zeros[0:x.shape[0], :] = x
    return zeros

def create_x_plot_matrix(x):
    zeros = np.zeros((MAX_LEN, x.shape[1]))
    x = x.values
    x = x[:(MAX_LEN), 0:]
    zeros[0:x.shape[0], :] = x
    return zeros


def create_y_matrix(y):
    zeros = np.zeros((MAX_LEN, y.shape[1] - 4))
    y = y.values
    y = y[:, 4:]
    y = y[:MAX_LEN, :]
    zeros[:y.shape[0], :] = y
    return zeros

def make_3d_tensor_slices(GAP_TIME,X_tensor, Y_tensor, lengths , staticCol , static=np.nan,XScols=np.nan , static_to_keep = np.nan ,frameOffset=0):

    num_patients = X_tensor.shape[0]
    timesteps = X_tensor.shape[1]
    num_features = X_tensor.shape[2]
    X_tensor_new = np.zeros((lengths.sum(), SLICE_SIZE, num_features + 1))
    Y_tensor_new = np.zeros((lengths.sum()))

    current_row = 0
    PdxArr=[]

    for patient_index in range(num_patients):
        PdxArr.append([patient_index,current_row])
        x_patient = X_tensor[patient_index]
        y_patient = Y_tensor[patient_index]
        length = np.minimum(lengths[patient_index],y_patient.shape[0])
        #length = lengths[patient_index]

        if  do_pred_only!=1:
            timestampRang=length - PREDICTION_WINDOW - GAP_TIME - SLICE_SIZE
        else:
            timestampRang=1


        for timestep in range(timestampRang):
            if do_pred_only!= 1:
                x_window = x_patient[timestep:timestep + SLICE_SIZE]
            else:
                x_window = x_patient[ min (length,x_patient.shape[0]) - SLICE_SIZE :min (length,x_patient.shape[0]) ]
                pID = static.iloc[patient_index:patient_index+1,1:2].index.values[0][1]


                #fi = pd.read_csv('/home/icu/Desktop/mimic/MIMIC_Extract_tsg/src/Vent/_OnSet_Belinson/_ADAPTED_22-12-2020_1026-S6-G6-P4/FeatureImportance/_ADAPTED_22-12-2020_1026-S6-G6-P4__AGG_Sum_Importance_OnSet_Gap_6h_feature_importance.csv')
                #xx =
            # Remember To RELOCATE <Current supervised Y > on Features table and not on STATIC table !!!!  !!! Roy
            y_window = y_patient[timestep:timestep + SLICE_SIZE]
            ###############ROY ONSET ONLY
            if max(y_window) > 0 and do_pred_only!=1  :
                continue
            ##############
            x_window = np.concatenate((x_window[0:,0:staticCol-1], np.expand_dims(y_window, 1),x_window[0:,staticCol-1:]), axis=1)
            if do_pred_only==1:
                result_window = y_patient[y_patient.shape[0] - PREDICTION_WINDOW:y_patient.shape[0]]
            else:
                result_window = y_patient[
                            timestep + SLICE_SIZE + GAP_TIME:timestep + SLICE_SIZE + GAP_TIME + PREDICTION_WINDOW]

            result_window_diff = set(np.diff(result_window))
            # if 1 in result_window_diff: pdb.set_trace()
            if do_pred_only != 1:
                gap_window = y_patient[timestep + SLICE_SIZE:timestep + SLICE_SIZE + GAP_TIME]
            else:
                gap_window = y_patient[timestep :timestep + SLICE_SIZE]

            gap_window_diff = set(np.diff(gap_window))

            ###############ROY ONSET ONLY
            #if max(gap_window) > 0:
                #continue
            ##############


            # print result_window, result_window_diff

            if OUTCOME_TYPE == 'binary':
                if max(gap_window) == 1:
                    result = None
                elif max(result_window) == 1:
                    result = 1
                elif max(result_window) == 0:
                    result = 0
                if result != None:
                    X_tensor_new[current_row] = x_window
                    Y_tensor_new[current_row] = result
                    current_row += 1

            else:
                if 1 in gap_window_diff or -1 in gap_window_diff:
                    result = None
                elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 0):
                    result = CHUNK_KEY['CONTROL']
                elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 1):
                    result =  CHUNK_KEY['ONSET']
                elif 1 in result_window_diff:
                    result = CHUNK_KEY['ONSET']
                elif -1 in result_window_diff:
                    result = None
                else:
                    result = None

                if result != None or do_pred_only==1:
                    X_tensor_new[current_row] = x_window
                    Y_tensor_new[current_row] = result
                    current_row += 1  #pred only
                if do_pred_only==1:
                    break

    X_tensor_new = X_tensor_new[:current_row, :, :]
    Y_tensor_new = Y_tensor_new[:current_row]

    # Remember To RELOCATE <Current supervised Y > LAST INDEX ON X on Features table and not on STATIC table !!!!  !!! Roy
    return X_tensor_new, Y_tensor_new ,PdxArr


def remove_duplicate_static(x,time_series_col):
    x_static = x[:, 0, time_series_col:x.shape[2] ]
    x_timeseries = np.reshape(x[:, :, :time_series_col], (x.shape[0], -1))
    x_int = x[:, :, time_series_col -1]
    x_concat = np.concatenate((x_static, x_timeseries, x_int), axis=1)
    return x_concat

