
import numpy as np
import pandas as pd
import yaml
import glob
import seaborn as sns
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
mor = [["solute", "liquid", "solid"][i] for i in mor_class]

max_num = args["max_num"]
min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
val_fold = args["val_fold"]
data_dir = args["data_dir"]
file_name = args["file_name"]


plt.figure (figsize = (8, 8))

test= pd.read_csv(f'./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/total_result_val_{min_num}_{max_num}.csv')

y_pred=test["preds"]
y_test=test['actual']

#平均のROC曲線を描く
matrix = confusion_matrix(y_test, y_pred,)
def make_cm_total(matrix, mor):

        #データフレーム生成
        cm_total = pd.DataFrame(matrix,
            columns=mor, index= mor)
        return cm_total


cm_total = make_cm_total(matrix, mor)
print(cm_total)
plt.figure(figsize = (10, 10))
with plt.style.context({'axes.labelsize':80,
                        'xtick.labelsize':24,
                        'ytick.labelsize':24}):
    sns.heatmap(cm_total,annot=True,cmap='Blues',fmt="d",annot_kws={'size':36}, cbar=False)
plt.ylabel('Answer', fontsize=30, labelpad = 15)
plt.xlabel('Prediction', fontsize=30, labelpad = 15)
plt.title("Confusion matrix (Chin)", fontsize = 32 , pad = 16)

plt.savefig(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/confusion_matrix.png")
# print("classification_report: ")
# print(classification_report(y_test, y_pred))
# print(matrix)

# report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict = True))

# y_prob=test["liquid"]
# y_test=test['actual']
# precision_ave, recall_ave, thresholds_ave = metrics.precision_recall_curve(y_test, y_prob)
# auc_ave = metrics.auc(recall_ave, precision_ave)

# fpr_ave, tpr_ave, thresholds_ave = metrics.roc_curve(y_test,y_prob)
# roc_auc = metrics.auc(fpr_ave,tpr_ave)

# # df_report = pd.DataFrame (report_df["macro avg"]).T
# df_report = pd.DataFrame()
# df_report.loc[0, "roc_auc"] = roc_auc
# df_report["pr_auc"] = auc_ave
# df_report["accuracy"] = metrics.accuracy_score(y_test, y_pred)
# # df_report = df_report.reindex(columns = ["roc_auc", "pr_auc", "precision", "recall", "accuracy", "f1-score"])
# df_report["precision"] = metrics.precision_score(y_test, y_pred)
# df_report["recall"] =metrics.recall_score(y_test, y_pred)
# df_report["f1_score"] =metrics.f1_score(y_test, y_pred)

# df_report.to_csv(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/score.csv")

# print("acc: ", metrics.accuracy_score(y_test, y_pred))
# print("precision: ", metrics.precision_score(y_test, y_pred))
# print("recall: ", metrics.recall_score(y_test, y_pred))
