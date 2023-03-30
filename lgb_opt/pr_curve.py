import argparse
import numpy as np
import pandas as pd
import yaml
import glob
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut, cross_val_score, cross_validate
import lightgbm as lgb

import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
val_fold_strategy = args["val_fold_strategy"]
n_splits = args["n_splits"]
random_state = args["random_state"]

mor_class = np.array(args["mor_class"])
data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}

max_num = args["max_num"]
min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
val_fold = args["val_fold"]
data_dir = args["data_dir"]
file_name = args["file_name"]

fig = plt.figure(figsize=(10, 10))
file_list = []

file_list = sorted(glob.glob(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/pred_cutdata_val_{min_num}_{max_num}_fold*.csv"))
for i, filename in enumerate(file_list):
    test = pd.read_csv(filename, )

    y_pred=test['liquid']
    y_test=test['actual']
    #AUCを求める
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
    auc = metrics.auc(recall, precision)
    plt.plot(recall, precision,  color = "darkgrey",alpha = 0.3, lw = 0.9)

#平均のAUCを求める
test= pd.read_csv(f'./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/total_result_val_{min_num}_{max_num}.csv')

y_pred_ave=test["liquid"]
y_test_ave=test['actual']
precision_ave, recall_ave, thresholds_ave = metrics.precision_recall_curve(y_test_ave, y_pred_ave)
auc_ave = metrics.auc(recall_ave, precision_ave)
print(f"###### total, data = {min_num, max_num} precision/recall######")
# print(f"precision: ", precision_ave, )
# print(f"recall: ", recall_ave, )
print("pr_auc:", auc_ave)
#平均のROC曲線を描く
plt.plot(recall_ave, precision_ave, label=f'total (area = %.2f)'%auc_ave, color = "black", alpha = 1.0,  lw = 1.2)

plt.legend(loc = "lower right", fontsize = 18) 
plt.title(f'ROC {min_num}_{max_num} {fold_strategy}\n val_fold = {val_fold_strategy} (er = {early_stopping_num}, val_fold = {val_fold})',  fontsize = 24, pad = 10)
plt.xlabel('Recall', fontsize = 16)
plt.ylabel('Precision',  fontsize = 16)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.grid(True)
plt.show()
plt.savefig(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/pr_curve_opt_lgb_{file_name}{max_num}_{fold_strategy}.png")
