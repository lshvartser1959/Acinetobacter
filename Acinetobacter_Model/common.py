FILE = 'full' #full, or development (3000
PRE_PROCESS = True
GRID_SEARCH = False
HYP_SEARCH = False
if FILE == 'full':
    DATAFILE = '/out/bacteria_out/all_hourly_data.h5'
    #DATAFILE = '/mnt/data2/avivm/mimic_extract_output/all_hourly_data_1000_60_.h5'
    Xmerge = 'X_merge_full'
    Ymerge = 'Y_merge_full'
    rows_to_keep_merge = 'rows_to_keep_merge_full'
    static_merge = 'static_merge_full'

elif FILE == 'small':
    DATAFILE = '/out/bacteria_out/all_hourly_data_100_60_.h5'
    Xmerge = 'X_merge_small'
    Ymerge = 'Y_merge_small'
    rows_to_keep_merge = 'rows_to_keep_merge_small'
    static_merge = 'static_merge_small'

else:
    DATAFILE = '/out/bacteria_out/all_hourly_data_3000_60_.h5'

    Xmerge = 'X_merge_3000'
    Ymerge = 'Y_merge_3000'
    rows_to_keep_merge = 'rows_to_keep_merge_3000'
    static_merge = 'static_merge_3000'

EXPORT_IMPORT_PATH = '/media/data/jj_acinetobacter/'
#INTERVENTION = ['acinetobacter', 'klebsiella', 'pseudomonas','mrsa']
INTERVENTION = ['acinetobacter']
# RANDOM = 34000
NUM_CLASSES = 2

SLICE_SIZE = 6
# GAP_TIME = 24
PREDICTION_WINDOW = 4

TEST_SIZE = 0.25
MAX_LEN = 4500 #What is this?
OUTCOME_TYPE = 'binary' #'all'

RUN_ADD_FEATURES = True
READ_ROLLING_WINDOW = False

readstdlist = 0
frameOffset = 0
do_pred_only = 0
Comment = 'dev'
do_adapt = 0