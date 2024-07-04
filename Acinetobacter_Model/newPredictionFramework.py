# for checking out different gap sizes
from methods import *
from AUCROC import *

#path2 = "/media/data/jj_acinetobacter/medical/src/Acinetobacter_Model/"
path2 = "/mnt/data2/"

def prepare_x_y1():
    # Load data from DATAFILE
    X, Y, static, rows_to_keep, outcomes = load_data(DATAFILE)
    # remain only those that have rows_to_keep, positives or negatives
    X2, outcomes, Y, static, rows_to_keep = filter_data(X, Y, static, rows_to_keep, outcomes)
    # add time of intervention to static
    static, lengths = add2static(Y, static)

    # run simple_imputer
    X_clean = imputation(X2)

    # add interventions
    outcomes.columns = pd.MultiIndex.from_product([outcomes.columns, ['mean']])
    X_clean = X_clean.join(outcomes, how='inner')
    X_clean.index = X_clean.index.remove_unused_levels()  # X2 replaces X. Keeping it as seperate variable for debugging

    # normalization of features
    X_std, XScols = normalization(X_clean)
    # add statics to the data
    X_merge, time_series_col, static_to_keep = merge(X_std, static)
    return X_merge, Y, rows_to_keep, static

def main():
    # Only run when the data has not yet been pre-processed
    if PRE_PROCESS:
        X_merge, Y, rows_to_keep, static = prepare_x_y1()
        X_merge.to_csv(path2 + Xmerge)
        Y.to_csv(path2 + Ymerge)
        rows_to_keep.to_csv(path2 + rows_to_keep_merge)
        static.to_csv(path2 + static_merge)
    else:
        # Only run when the data just has been pre-processed
        static = pd.read_csv(path2 + static_merge).set_index(['subject_id', 'hadm_id', 'icustay_id'])
        if RUN_ADD_FEATURES:
            X_merge=pd.read_csv(path2+ Xmerge).set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
            print('Merging')
            Y=pd.read_csv(path2 + Xmerge).set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
            rows_to_keep=pd.read_csv(path2 + rows_to_keep_merge).set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
            Y.columns = INTERVENTION
    test_scores = {}

    # It take a long time to run this double cycle than at first run gap_times = [0] and random_seeds = [1]
    random_seeds = [1,53,312, 507, 128, 10, 2000, 889, 27, 182, 343, 20, 89, 25, 89, 456, 73, 761, 12, 805]
    gap_times = [2, 4, 6, 12, 18, 24, 48, 96]
    #gap_times = [6]
    #random_seeds = [761]
    # import the metrics class
    from sklearn import metrics
    # import the class
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

    for gap_time in gap_times:
        scores_per_seed = []
        if READ_ROLLING_WINDOW:

            X=pd.read_csv(path2 + 'X_rolled_18_full.csv')
            Y = pd.read_csv(path2 + 'Y_rolled_18_full.csv')
            X=X.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
            Y=Y.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
            X=X.drop("index",axis=1)
            Y=Y.drop("index",axis=1)
            u=~np.isnan(X["alanine aminotransferase mask"])
            X=X[u]
            Y = Y[u]
        print('Creating Rolling Window')
        #rk = rows_to_keep


        if RUN_ADD_FEATURES:
            # Create window 6 hours measurements, gap_time hours gap, 4 hours predict
            X, Y, rows_to_keep = create_long_window(X_merge, Y, rows_to_keep, gap_size=gap_time, path=path2)
            #X, Y, rows_to_keep = create_rolling_window(X_merge, Y, rows_to_keep, gap_size=gap_time)
            u = ~np.isnan(X["alanine aminotransferase mask"])
            X = X[u]
            Y = Y[u]
            a_file = open('/mnt/data2/' + "X.pkl", "wb")
            pickle.dump(X, a_file)
            a_file.close()
            b_file = open('/mnt/data2/' + "Y.pkl", "wb")
            pickle.dump(Y, b_file)
            b_file.close()

            print('long finished')

            # ATTEMPTS TO ADD OTHER FEATURES, NO SUCCESS

            # X, Y, rows_to_keep = create_rolling_window(X_merge, Y, rows_to_keep, gap_size=gap_time)
            # u = ~np.isnan(X["alanine aminotransferase mask"])
            # X = X[u]
            # Y = Y[u]
            # X.to_csv(path2 + 'X.csv')
            # Y.to_csv(path2 + 'Y.csv')
            # print('mean finished')
            # X1, rows_to_keep = create_long_shifted_vectors(X_merge, rk, gap_time, 'var')
            # X1 = X1[u]
            # X1.to_csv(path2 + 'X1.csv')
            # print('var finished')
            # X2, rows_to_keep = create_long_shifted_vectors(X_merge, rk, gap_time, 'min')
            # X2 = X2[u]
            # X2.to_csv(path2 + 'X2.csv')
            # print('min finished')
            # X3, rows_to_keep = create_long_shifted_vectors(X_merge, rk, gap_time, 'max')
            # X3 = X3[u]
            # X3.to_csv(path2 + 'X3.csv')
            # print('max finished')
            # X4, rows_to_keep = create_long_shifted_vectors(X_merge, rk, gap_time, 'skew')
            # X4 = X4[u]
            # X4.to_csv(path2 + 'X4.csv')
            # print('skew finished')
            # X5, rows_to_keep = create_long_shifted_vectors(X_merge, rk, gap_time, 'kurt')
            # X5 = X5[u]
            # X5.to_csv(path2 + 'X5.csv')
            # print('kurt finished')
        else:
            a_file = open('/mnt/data2/' + "X.pkl", "rb")
            X = pickle.load(a_file)
            a_file.close()
            b_file = open('/mnt/data2/' + "Y.pkl", "rb")
            Y = pickle.load(b_file)
            b_file.close()

        # ATTEMPTS TO ADD OTHER FEATURES, NO SUCCESS

        #     X1=pd.read_csv(path2 + 'X1.csv')
        #     X1=X1.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
        #     X1=X1.drop("index",axis=1)
        #     X1.columns = [str(col) + '_var' for col in X1.columns]
        #
        #     X2=pd.read_csv(path2 + 'X2.csv')
        #     X2=X2.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
        #     X2=X2.drop("index",axis=1)
        #     X2.columns = [str(col) + '_min' for col in X2.columns]
        #
        #     X3=pd.read_csv(path2 + 'X3.csv')
        #     X3=X3.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
        #     X3=X3.drop("index",axis=1)
        #     X3.columns = [str(col) + '_max' for col in X3.columns]
        #
        #     X4=pd.read_csv(path2 + 'X4.csv')
        #     X4=X4.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
        #     X4=X4.drop("index",axis=1)
        #     X4.columns = [str(col) + '_skew' for col in X4.columns]
        #
        #     X5=pd.read_csv(path2 + 'X1.csv')
        #     X5=X5.reset_index().set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
        #     X5=X5.drop("index",axis=1)
        #     X5.columns = [str(col) + '_kurt' for col in X5.columns]
        #
        # X = pd.concat([X,X1,X2,X3,X4,X5], axis=1)
        # X.to_csv(path2 + 'XL.csv')
        # print('concat finished')

        # add new static columns of Ido
        static_cols = ['vent', 'nivdurations', 'colloid_bolus', 'crystalloid_bolus',
                       'nausea', 'vomit', 'drain or tube', 'tracheostomy', 'drain', 'urine cath'] + [
                          'past_admissions', 'past_admissions_icu', 'prev_surgeries', 'prev_surgeries_mv']
        static_cols = [i for i in static_cols if i in static.columns]
        static_cols = [i for i in static_cols if i not in X.columns]
        X = X.join(static[static_cols])  # , 'ethnicity', 'first_careunit']]
        X['hours_in'] = X.index.get_level_values('hours_in')
        # X.to_csv(path2 + 'X_rolled_18_full.csv')
        # Y.to_csv(path2 + 'Y_rolled_18_full.csv')
        ii=0

        auc = [i for i in range(len(random_seeds))]
        for random_seed in random_seeds:
            # split data to train, validation, test
            print(f'Gap time = {gap_time}, Random seed = {random_seed}')
            train_ids, val_ids, test_ids = train_test_val(static, random_seed)
            print('Splitting data')
            # split by 3 levels of multiple index
            x_train, x_val, x_test, y_train, y_val, y_test = tvt_split(X, Y, train_ids, val_ids, test_ids)
            x_train = x_train.fillna(0)
            x_test = x_test.fillna(0)

            print('Training')
            LOGREG = True
            if LOGREG:
                # instantiate the model (using the default parameters)
                logreg = LogisticRegression(random_state=random_seed)

                # fit the model with data
                logreg.fit(x_train, y_train)

                y_pred = logreg.predict(x_test)

                cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
                y_pred_proba = logreg.predict_proba(x_test)[::, 1]
                fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
                test_score = metrics.roc_auc_score(y_test, y_pred_proba)
                scores_per_seed.append(test_score)
                # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
                # plt.legend(loc=4)
                # plt.show()
                # aa=0
                feature_impo = logreg.coef_
                ii+=1
            else:

                if random_seed == 1:
                    model, feature_impo, test, pred, test_score = training_testing(x_train, x_val, x_test, y_train.values.ravel(),
                                                                                   y_val, y_test.values.ravel(), 'no model',
                                                                                   random_seed, gap_time,
                                                                                   grid_search=False,
                                                                                   hyp_search=True,n_trials=5)  # Set hyp_search to true for optuna. Grid search to true for small
                    model1 = model
                else:
                    # training testing including Optuna or grid search for hyperparameters
                    model, feature_impo, test, pred, test_score = training_testing(x_train, x_val, x_test, y_train.values.ravel(),
                                                                                   y_val, y_test.values.ravel(), model1,
                                                                                   random_seed, gap_time,
                                                                                   grid_search=False,
                                                                                   hyp_search=False,n_trials=5)  # Set hyp_search to true for optuna. Grid search to true for small

                    ROC_AUC_CONF_MATRIX = True
                    if ROC_AUC_CONF_MATRIX:
                        # ROC - AUC,confusion matrix
                        pred1 = pred[0].values
                        pred0 = pred[1].values
                        pred[0] = pred0
                        pred[1] = pred1
                        # calc ROC-AUC, confusion matrix
                        Gntbok1(pred, test, gap_time, SLICE_SIZE, PREDICTION_WINDOW,gap_time,random_seed)

                    scores_per_seed.append(test_score)
        test_scores[gap_time] = scores_per_seed
        fi = pd.DataFrame(abs(feature_impo.T), index=X.columns)
        fi.to_csv('/mnt/data2/'+'feature_impoLR_'+ str(gap_time)+'_int.csv')
        pd.DataFrame(test_scores).to_csv('/mnt/data2/'+'test_scores_gap_timesLR_'+ str(gap_times)+'_int.csv')

    print(test_scores)
    fi = pd.DataFrame(abs(feature_impo.T), index=X.columns)
    fi.to_csv('/mnt/data2/'+'feature_impo'+'_int.csv')
    pd.DataFrame(test_scores).to_csv('/mnt/data2/'+'test_scores_gap_timesLR'+'_int.csv')

if __name__ == '__main__':
    main()
