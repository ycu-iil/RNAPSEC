
"Uniprotidごとのデータ数"
#データセットのprotein_sequenceからuniprotidを抽出、融合タンパク質はfusionに統一
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

"各論文の収録データ数を棒グラフで図示"
#RNAPhaSep (Curated)とRNAPSEC (Curated)
with open("../main_exdata_1212/config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
mor_class = np.array(args["mor_class"])

"データの用意"
df_default = pd.read_csv("/home/chin/rnaphasep_vs/clf_mor/model/model_0222/dataset_0222/original/default_dataset_0221.csv")
df_default = df_default.drop(df_default.columns[df_default.columns.str.contains("Unnamed")], axis = "columns")
df_default = df_default.rename(columns = {"index": "rnap_index"})
print(df_default.columns[df_default.columns.str.contains("classi")])

df_chin = pd.read_csv("../data/chin_all_col.csv", index_col=False)
df_chin = df_chin.rename(columns = {"index": "ini_idx"})
df_chin = df_chin.rename(columns = {"ini_idx": "rnap_index"})
df_chin = pd.merge(left = df_chin, right = df_default, how = "left", on = "rnap_index")
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_chin = df_chin[df_chin[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
df_chin_all_col = pd.merge(left = df_chin, right = df_default, how = "left", on = "rnap_index")
print(df_chin.columns)
df_chin_all_col.__getitem__("rna_classification_x").__setitem__(df_chin_all_col[df_chin_all_col.rna_classification_x == "irregular RNA;|tRNA"].index, "else")


chin_rna_clf_name = df_chin_all_col.rna_classification_x.value_counts().index
chin_rna_clf_counts = df_chin_all_col.rna_classification_x.value_counts()

df_prepro = pd.read_csv("../data/prepro_all_col.csv", index_col=False)
df_prepro = df_prepro[df_prepro[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
prepro_rna_clf_name = df_prepro.rna_classification.value_counts().index
prepro_rna_clf_counts = df_prepro.rna_classification.value_counts()

fig = plt.figure(figsize = (8, 6))
plt.bar(x = chin_rna_clf_name, height = chin_rna_clf_counts, label = "RNAPSEC (Curated)", alpha = 0.8)
plt.bar(x = prepro_rna_clf_name, height = prepro_rna_clf_counts, label = "RNAPhaSep (Curated)", alpha = 0.8)
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend(fontsize = 18)
plt.title("Number of data for each RNA type", fontsize = 22)
plt.ylabel("Number of data", fontsize = 18)
plt.xlabel("RNA type", fontsize = 18)
fig.savefig("./plt_database/fig7b_rnatype.png")
print("end")