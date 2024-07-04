import time , os
import pickle

def Gntbok1(pf, tf, GAP, SLICE_SIZE, PREDICTION_WINDOW, GAP_SIZE, RANDOM):
    # %%

    from scipy import interp
    import numpy as np
    import pandas as pd
    from itertools import cycle
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.metrics import roc_curve, auc, recall_score, confusion_matrix, precision_score, accuracy_score
    # from scipy import interp
    from sklearn.metrics import roc_auc_score
    import seaborn as sns
    import matplotlib.pyplot as plt

    # import plotly.plotly as py
    # import plotly.graph_objs as go

    # %%

    # y_pred = (model.predict_proba(x_flat_test)[:,1] >=0.3).astype(bool) #set threshold

    # %%

    # vaso DATA
    # PRED_PATH = "/home/roy/tmp/mimic/MIMIC_Extract_tsg/src/Vent/Aucroc/12-04-2020_2227ourly_datah5__XGB_Y_pred.csv"
    # TEST_PATH = "/home/roy/tmp/mimic/MIMIC_Extract_tsg/src/Vent/Aucroc/12-04-2020_2227ourly_datah5__XGB_Y_test.csv"


    print('ok!')

    # %%

    y_test = tf
    y_pred = pf
    print('ok!')

    # %%

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)


    # %% md

    # best threshold

    # %%

    y_score = y_pred.values
    y_true = y_test.values

    lw = 2
    #ONSET
    n_classes = 1
    #n_classes = y_true.shape[1]

    fpr = dict()
    tpr = dict()
    thr = dict()
    roc_auc = dict()
    _auc=dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thr[i] = roc_curve(y_true[:, i], y_score[:, i])
        _auc[i] =roc_auc_score(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # # # Compute micro-average ROC curve and ROC area
    # # fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    # # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])

    sns.set(font_scale=0.9)

    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of \nclass {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))

    #ONSETONLY
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class OnSet \n(area = {0:0.2f})'
                       ''.format(roc_auc[i]))

    #plt.title('Confusion matrix of', fontsize=20)

    # if (SLICE_SIZE != -1):
    #     os.environ['SLICE_SIZE'] = SLICE_SIZE
    # if (PREDICTION_WINDOW != -1):
    #     os.environ['PREDICTION_WINDOW'] = PREDICTION_WINDOW

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.10])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if (SLICE_SIZE != -1):
        plt.title('ROC GAP_'+str(GAP) +' Sl_'+str(SLICE_SIZE ) +' Pr_'+str(PREDICTION_WINDOW))
    else:
        plt.title ('ROC GAP_' + str (GAP) + ' Sl_' + str (os.environ['SLICE_SIZE']) + ' Pr_' + str (os.environ['PREDICTION_WINDOW']))
    plt.legend(title='Parameter where:')
    plt.legend(bbox_to_anchor=(1, 0., 0.5, 0.5))
    # plt.legend(loc="lower right")
    time.sleep(4)
    plt.show()


    print('OK')

    # %%



    # %%



    # %%



    # %%

    # plt.plot(fpr[0], tpr[0])
    # time.sleep(4)
    # plt.show()


    # %%

    # plt.plot(fpr[0], thr[0])
    # time.sleep(4)
    # plt.show()



    # %%

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return {'idx':idx, 'val': array[idx]}

    # %%

    # FP=(1- _auc[0])
    # print ('\nBest FP cut :'+ str(FP))
    #
    # print (find_nearest(fpr[0], FP)[0])
    #
    # # %%
    #
    # tpr[0][find_nearest(fpr[0], 0.2)[0]]
    #
    # FP = 1-tpr[0][find_nearest(fpr[0], 0.2)[0]]

    # %%

    # print (thr[0][find_nearest(fpr[0],0.2)[0]] ) # ~0.004576941

    # %%

    best_FP_idx= find_nearest(fpr[0], 0.2)['idx']
    print ('True Positive rate at bestFP index:',tpr[0][best_FP_idx])

    if tpr[0][best_FP_idx] > 0.95:
        while tpr[0][best_FP_idx] > 0.95:
            best_FP_idx -=1
    elif tpr[0][best_FP_idx] > 0.9:
        while tpr[0][best_FP_idx] > 0.9:
            best_FP_idx -= 1

    y_pred1 = y_pred >= thr[0][best_FP_idx]

    print('\nBest threshold :', thr[0][best_FP_idx]) # ~0.004576941

    #(prob - thresh)/(prob_max - thresh) for positive, (thresh - prob)/(thresh - prob_min)
    # minPredProba=np.min(y_pred[0:, 0])
    # maxPredProba=np.max(y_pred[0:, 0])
    # minPredProba = np.min(y_pred.iloc[:, 0])
    # maxPredProba = np.max(y_pred.iloc[:, 0])
    #
    #
    # change=0
    # if (len (PdxArr)):
    #     with open('MinMaxProba', mode='rb') as f:
    #         minmax = pickle.load(f)
    #
    #     #LOAD MINMAX
    #     currentMinProba = minmax[0] #load it
    #     currentMaxProba = minmax[1] #load it
    #
    #     if minPredProba < currentMinProba:
    #         change=1
    #         currentMinProba= minPredProba
    #
    #     if maxPredProba > currentMaxProba:
    #         change=1
    #         currentMaxProba = maxPredProba
    #
    #     if change==1:
    #         minmax=[]
    #         minmax.append(currentMinProba)
    #         minmax.append(currentMaxProba)
    #
    #         with open('MinMaxProba', 'wb') as f:
    #             # Pickle the 'data' dictionary using the highest protocol available.
    #             pickle.dump(minmax, f, pickle.HIGHEST_PROTOCOL)




        # ptv = []
        # #iiXX=range(0,y_pred1[0:, 0])
        # iiXX = range(0, PdxArr.__len__())
        # for ii in iiXX:
        #     try:
        #         iipx = range(PdxArr[ii][1], PdxArr[ii+1][1])
        #     except:
        #         iipx = range(PdxArr[ii][1], y_pred1[0:, 0].shape[0] - 1)
        #
        #
        #     ptv.append([ii, 0])
        #     for jj in iipx:
        #         if y_pred1[0:, 0][jj]==1:
        #             ptv[ii][1]=1
        #             break;
        # ptvdf = pd.DataFrame(ptv,columns =[ "PatientStaticIdx", "Vent6hours" ])
        # ptvdf.to_csv('/media/data/res/pred_res.csv')
    # ptv = []
    # # iiXX=range(0,y_pred1[0:, 0])
    # iiXX = range(0, PdxArr.__len__())
    # for ii in iiXX:
    #     try:
    #         iipx = range(PdxArr[ii][1], PdxArr[ii + 1][1])
    #     except:
    #         iipx = range(PdxArr[ii][1], y_pred1.iloc[:, 0].shape[0] - 1)
    #
    #     ptv.append([ii, 0])
    #     for jj in iipx:
    #         if y_pred1.iloc[:, 0][jj] == 1:
    #             ptv[ii][1] = 1
    #             break;
    # ptvdf = pd.DataFrame(ptv, columns=["PatientStaticIdx", "Vent6hours"])
    # ptvdf.to_csv('/media/data/shua_acinetobacter/medical/src/Acinetobacter_Model_Leonid/pred_res.csv')
    #
    #
    #


    # %%

   # y_pred1

    # %%

    confusion_matrix(y_true[0:, 0], y_pred1.iloc[:, 0])

    # %%

    if (SLICE_SIZE != -1):
        plt.title('Confusion matrix OnSet GAP '+str(GAP) +' Sl_'+str(SLICE_SIZE ) +' Pr_'+str(PREDICTION_WINDOW), fontsize=19)
    else:
        plt.title('Confusion matrix OnSet GAP '+str(GAP)+' Sl_'+str(os.environ['SLICE_SIZE'] ) +' Pr_'+str(os.environ['PREDICTION_WINDOW'] ), fontsize=19)

    sns.set(font_scale=1.4)

    sns.heatmap(confusion_matrix(y_true[0:, 0], y_pred1.iloc[:, 0]), cmap="BuPu", annot=True, fmt="d",
                annot_kws={"size": 16})
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('True', fontsize=18)
    time.sleep(4)
    plt.show()



    # %%

    if (SLICE_SIZE != -1):
        plt.title('ONSET GAP '+str(GAP) +' Sl_'+str(SLICE_SIZE ) +' Pr_'+str(PREDICTION_WINDOW))
    else:
        print ('ONSET GAP '+str(GAP)+' Sl_'+str(os.environ['SLICE_SIZE'] ) +' Pr_'+str(os.environ['PREDICTION_WINDOW'] ))

    print('\n roc_auc_score ', roc_auc_score(y_true[0:, 0], y_pred.iloc[:, 0]))
    with open('results.txt', 'w') as f:
        f.write(f'Gap size = {GAP_SIZE}, Random seed = {RANDOM}, Result = {roc_auc_score(y_true[0:, 0], y_pred.iloc[:, 0])}')

    print('\n recall_score ', recall_score(y_true[0:, 0], y_pred1.iloc[:, 0]))

    # %%

    print('\n precision_score ', precision_score(y_true[0:, 0], y_pred1.iloc[:, 0]))

    # %%

    print('\n accuracy_score ', accuracy_score(y_true[0:, 0], y_pred1.iloc[:, 0]))

    # %%

    print('\n roc_auc_score_macro ', roc_auc_score(y_true[0:, 0], y_pred.iloc[:, 0], average='macro'))

    pass


if __name__ == '__main__':
    pass
