
from IPython.display import display
import seaborn as sns
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

min_num = args["min_num"]
max_num = args["max_num"]
early_stopping_num = args["early_stopping"]
val_fold = 5
data_dir = args["data_dir"]
file_name = args["file_name"]



def plt_fi(df_fi, db_name):
    fig_fi = plt.figure()
    df_fi["mean"].sort_values( ascending=True).tail(20).plot.barh(figsize=(10,8),label="methods = split", fontsize=20)
    plt.xlabel("Feature importance",  fontsize=20)
    plt.ylabel("feature",  fontsize=20)
    plt.legend(loc="lower right", fontsize=18)
    plt.grid()
    plt.title(f"{db_name}\n fold strategy = {fold_strategy}",  fontsize=24)
#     plt.title(f"{db_name} ",  fontsize=24)
    fig_fi.savefig(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/feature_importances_{file_name}_mor{len(mor_class)}_{fold_strategy}.png")
    return 

# df_name = ["chin", "prepro"]
#files = data_dir + df_name[0] + '.csv'
mor = [["solute", "liquid", "solid"][i] for i in mor_class] #予測対象の形態の名称リスト

import pickle
fis = {}
features =  pd.read_csv( data_dir + file_name + '.csv').columns.drop([group_label, target_label])
features = features.str.replace("_", " ")
features = features.str.replace("rna rna", "rna")
features = features.str.replace("protein", "(protein)")
features = features.str.replace("rna", "(RNA)")
features = features.str.replace(" conc", " conc.")
features = features.str.replace("log", "(log)")
features = features.str.replace("temp", "Temperature")
features = features.str.replace("ionic strength", "Ionic strength")

df = pd.read_csv( data_dir + file_name + '.csv')
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df = df[df[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
max_num = df.shape[0]
for i in range(n_splits):
    with open(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/model_list_{file_name}.pickle", 'rb') as web:
        model = pickle.load(web)
    fis[i] = model[i].feature_importances_
df_fi=pd.DataFrame(fis, index=features)
df_fi["mean"]=df_fi.mean(axis="columns")
df_fi.to_csv(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/feature_importances_{file_name}_mor{len(mor_class)}_{fold_strategy}.csv", index = False)
plt_fi(df_fi, file_name)