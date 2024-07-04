import sys,os

import pandas as pd

sys.path.insert(0,'/media/data/jj_acinetobacter/medical/src/Acinetobacter_Model')
from methods import *

path = '/media/data/jj_acinetobacter/'
x_train = pd.read_csv(path + 'x_train_all.csv').set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
y_train = pd.read_csv(path + 'y_train_all.csv').set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
x_test = pd.read_csv(path + 'x_test_all.csv').set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
y_test = pd.read_csv(path + 'y_test_all.csv').set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])

model, feature_impo, test, pred = training_testing(x_train, 'x_val_placeholder', x_test, y_train, 'y_val_placeholder',y_test)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train, plot_type="bar")
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

from xgboost import XGBClassifier
for i in [10,20,50,100,200,len(x_train.columns)]:
    top10_vars = x_train.columns[np.argsort(np.abs(shap_values).mean(0))][:i]
    model_small = XGBClassifier()
    model_small.fit(x_train[top10_vars],y_train)
    print(i,roc_auc_score(y_test,model_small.predict_proba(x_test[top10_vars])[:,1]))
#After here trying to graph, didn't work
# def qool_cut(s, n_bins=50):
#     # cut into quantiles and represent bins using left integers
#
#     categories, edges = pd.qcut(s, n_bins, retbins=True, labels=False, duplicates='drop')
#
#     return categories.apply(lambda x: try_index(x, edges[0:]))
# def try_index(x, y):
#     try:
#
#         return y[int(x)]
#
#     except:
#
#         return None
#
# def graph_a_var(data, var_name1, q1=0, round1=0, min_number=200, scatter=False, target_var='surrender',
#                 count_var="EXP_ID_NUMBER"):
#     import plotly.express as px
#     if q1 > 0:
#         data[var_name1 + '_cut'] = qool_cut(data[var_name1], q1).round(round1)
#         var_name1 = var_name1 + '_cut'
#     df5 = data.groupby([var_name1]).agg({target_var: 'mean', count_var: 'nunique'}).reset_index()
#     df5 = df5[df5[count_var] > min_number]
#     fig = px.line(df5, x=var_name1, y=target_var, hover_data=[count_var])
#     fig.show()
# top10_vars =['alanine aminotransferase mask', 'pulmonary artery pressure mean mask',
#        'cholesterol hdl mask', 'cholesterol mask',
#        'systolic blood pressure time_since_measured', 'chloride urine mask',
#        'lactate dehydrogenase mask', 'post void residual mask',
#        'chloride mask', 'cholesterol ldl mask']
# data2 = pd.concat([x_test,y_test],axis=1)
# for i in top10_vars:
#     graph_a_var(data2, i, target_var='0', count_var='age_1', q1=10)
#
