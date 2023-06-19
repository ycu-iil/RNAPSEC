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
df_chin = pd.read_csv("../data/chin_all_col.csv", index_col=False)
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_chin = df_chin[df_chin[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)

df_prepro = pd.read_csv("../data/prepro_all_col.csv", index_col=False)
df_prepro = df_prepro[df_prepro[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)

df_chin["uniprotid"] =  df_chin.protein_sequence.str.split("|", 2, expand = True)[1]
df_chin["uniprotid"][ df_chin.uniprotid.str.contains("Linker is missing", na = False)] = "Fussion"
df_chin["uniprotid"][df_chin.uniprotid ==''] = "Fussion"
df_chin["uniprotid"][df_chin.uniprotid =='length=282aa'] = "Fussion"
df_chin["uniprotid"][df_chin.uniprotid.isna()]= "Repeat"

df_prepro["uniprotid"] =  df_prepro.protein_sequence.str.split("|", 2, expand = True)[1]
df_prepro["uniprotid"][ df_prepro.uniprotid.str.contains("Linker is missing", na = False)] = "Fussion"
df_prepro["uniprotid"][df_prepro.uniprotid ==''] = "Fussion"
df_prepro["uniprotid"][df_prepro.uniprotid =='length=282aa'] = "Fussion"
df_prepro["uniprotid"][df_prepro.uniprotid.isna()]= "Repeat"

#作図
fig = plt.figure(figsize = (8, 6))
y = df_chin.uniprotid.value_counts()
x = df_chin.uniprotid.value_counts().index
plt.bar(x , height = y, label= "RNAPSEC (Curated)", alpha = 0.8)
y_prepro = df_prepro.uniprotid.value_counts()
x_prepro = df_prepro.uniprotid.value_counts().index
plt.bar(x_prepro, y_prepro,label = "RNAPhaSep (Curated)", alpha = 0.8)
plt.legend(fontsize = 18)
plt.xticks(rotation= 90)
plt.xlabel("Uniprot ID", fontsize = 18)
plt.ylabel("Number of data", fontsize = 18)
plt.title("Number of data for each protein", fontsize = 22)
plt.grid()
fig.savefig("./plt_database/fig7a_uniprotid_bar.png")
print("end")