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
df_prepro = pd.read_csv("../data/prepro_all_col.csv", index_col=False)
#chin
files = "../data/chin.csv"
ml_chin = pd.read_csv(files, index_col=False) 
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
ml_chin = ml_chin[ml_chin[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
#prepro
files = "../data/prepro.csv"
ml_prepro = pd.read_csv(files, index_col=False) 
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
ml_prepro = ml_prepro[ml_prepro[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
print(ml_chin.shape, ml_prepro.shape) #修論:ml_chin= 1381, ml_prepro = 223

df_default = pd.read_csv("/home/chin/rnaphasep_vs/clf_mor/model/model_0222/dataset_0222/original/default_dataset_0221.csv")
df_default_component_1 = df_default[df_default.components_type == ("RNA + protein")]

#concat用に名前を揃える
def col_rename(df, before, after):
    df = df.rename(columns = {before: after})
    return df
# df_pmid = col_rename(df_pmid, "pmid", "default")
chin_pmid = pd.DataFrame(df_chin.pmidlink.value_counts())
chin_pmid = col_rename(chin_pmid, "pmidlink", "chin_pmid")
prepro_pmid = pd.DataFrame(df_prepro.pmidlink.value_counts())
prepro_pmid = col_rename(prepro_pmid, "pmidlink", "prepro_pmid")
default_component_1_pmid = pd.DataFrame(df_default_component_1.pmidlink.value_counts())
default_component_1_pmid = col_rename(default_component_1_pmid, "pmidlink", "default_component_1_pmid")

#plot用に全データセットのpdをconcat
df_pmid_2 = pd.concat([ chin_pmid, prepro_pmid, default_component_1_pmid], axis = "columns")

#作図
pmid_cp = df_pmid_2[(df_pmid_2.prepro_pmid.notna())|( df_pmid_2.chin_pmid.notna())|( df_pmid_2.default_component_1_pmid.notna())].sort_values(by = "chin_pmid", ascending = False)
pmid_cp =  pmid_cp.reset_index(drop= False)
fig = plt.figure(figsize = (30, 20))
plt.bar(x = pmid_cp.index, height = pmid_cp.chin_pmid, alpha = 0.5, label = "RNAPSEC(Curated)")
# plt.bar(x = pmid_cp.index, height = pmid_cp.default_component_1_pmid, alpha = 0.5, label = "RNAPhaSep(DB)")
plt.bar(x = pmid_cp.index, height = pmid_cp.prepro_pmid, alpha = 0.5, label = "RNAPhaSep(Curated)")
plt.legend(fontsize = 60, loc ="upper right")
ax = plt.gca()
plt.yticks(fontsize = 60)
plt.xticks(fontsize = 60)
plt.grid()
plt.xlabel("Article Number", fontsize = 60)
plt.ylabel("Data", fontsize = 60)
# ax.xaxis.set_visible(False)
plt.title (" Number of data collected from each article", fontsize = 80, pad =30)
fig.savefig("./plt_database/fig6b_num_data_each_journal.png")
