#学習データ(val, train)にsmoteを適応
import argparse
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
import yaml
import glob
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut, cross_val_score, cross_validate
import lightgbm as lgb
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize


#trainだけsmoteかける。validationとtestはかけない
#交差検証とROC曲線、valデータ付き (分割方法はconfig_.yamlで指定）、パラメーター調整なし
with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)
def labeling(df, label="pmidlink"):
    le = preprocessing.LabelEncoder()
    labels = df[label].unique()
    labels_id = le.fit_transform(df[label])
    df[f"{label}_label"] = labels_id 
    group_label = df[f"{label}_label"]
    return df, group_label

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
val_fold_strategy = args["val_fold_strategy"]
n_splits = args["n_splits"]
random_state = args["random_state"]
data_dir = args["data_dir"]

mor_class = np.array(args["mor_class"])
data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}
# smote_data = args["smote_data"]

min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
val_fold = args["val_fold"]

opt_name = "opt_auc"
#test_train, train_valの分割方法
def def_fold_strategy(fold_strategy, n_split = n_splits):
    if fold_strategy == "GroupKFold":
        kf = GroupKFold(n_splits = n_split)
    # elif fold_strategy == "StratifiedGroupKFold":
    #     kf = StratifiedGroupKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    elif fold_strategy == "KFold":
        kf = KFold(n_splits = n_split, shuffle = True, random_state = random_state)
    elif fold_strategy == "StratifiedKFold":
        kf = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = random_state)        
    elif fold_strategy == "LeaveOneGroupOut":
        kf = LeaveOneGroupOut()
    return kf

test_cv = def_fold_strategy(fold_strategy, n_split=n_splits)
# val_cv = def_fold_strategy(val_fold_strategy, n_split=val_fold)
val_cv = StratifiedKFold(shuffle=True, random_state=42, n_splits=5)

# chin
file_name = "chin"
max_num = 1381

files_path = data_dir + file_name 
files =f"{files_path}/sel_{file_name}_{min_num}_{max_num}.csv"
df_chin = pd.read_csv(files, index_col=False)

mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_chin = df_chin[df_chin[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
df_chin = df_chin.dropna()
df_chin = df_chin.reset_index(drop = True)
#X, y, groups
if ignore_features == False:  #省くカラムがない場合   
    X = df_chin.drop([target_label,group_label], axis = "columns").reset_index(drop = True).values #group込みのX, 
else:
    X = df_chin.drop([target_label, group_label, ignore_features], axis = "columns").reset_index(drop = True).values
y = df_chin[target_label].values
df_all_chin = pd.read_csv("../data/chin_all_col.csv", index_col = False)
df_all_chin= df_all_chin[df_all_chin[target_label].isin(mor_class)].reset_index(drop = True)
df_chin["aa_rna"] = df_all_chin.aa + df_all_chin.rna_sequence
df_chin["db"] = "chin"
print(df_all_chin.shape)
print(df_chin.shape)
with open(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/model_list_{file_name}.pickle", 'rb') as web:
    model_chin = pickle.load(web)

# prepro
file_name = "prepro"
max_num = 223

files_path = data_dir + file_name 
files =f"{files_path}/sel_{file_name}_{min_num}_{max_num}.csv"
df_prepro = pd.read_csv(files, index_col=False)

mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_prepro = df_prepro[df_prepro[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
df_prepro = df_prepro.dropna()
df_prepro = df_prepro.reset_index(drop = True)
#X, y, groups
if ignore_features == False:  #省くカラムがない場合   
    X = df_prepro.drop([target_label,group_label], axis = "columns").reset_index(drop = True).values #group込みのX, 
else:
    X = df_prepro.drop([target_label, group_label, ignore_features], axis = "columns").reset_index(drop = True).values
y = df_prepro[target_label].values
df_all_prepro = pd.read_csv("../data/prepro_all_col.csv", index_col = False)
df_all_prepro= df_all_prepro[df_all_prepro[target_label].isin(mor_class)].reset_index(drop = True)
df_prepro["aa_rna"] = df_all_prepro.aa + df_all_prepro.rna_sequence
df_prepro["db"] = "prepro"
with open(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/model_list_{file_name}.pickle", 'rb') as web:
    model_prepro = pickle.load(web)
print(df_all_prepro.shape)
print(df_prepro.shape)

#タンパク質・RNAの組み合わせごとにラベリング
df_aa_rna = pd.concat([df_chin.loc[:, ["aa_rna", "db"]], df_prepro.loc[:, ["aa_rna", "db"]]] , axis = "index")
df_aa_rna, aa_rna_label = labeling(df_aa_rna, "aa_rna")
df_prepro["ex_label"] = df_aa_rna[df_aa_rna.db == "prepro"].aa_rna_label
df_chin["ex_label"] = df_aa_rna[df_aa_rna.db == "chin"].aa_rna_label

def create_test_ex(df):
    protein_conc = np.linspace(df.protein_conc_log.min(), df.protein_conc_log.max()+1, 20)
    protein_conc.shape
    rna_conc = np.linspace(df.rna_conc_log.min()-0.1, df.rna_conc_log.max()+1, 20)
    rna_conc.shape
    #データセット中の該当するデータのタンパク質濃度とRNA濃度の分布の範囲内をカバーする濃度値を設定
    patterns = []
    for protein in protein_conc:
        for rna in rna_conc:
            patterns.append((protein,rna))
    df_add= pd.DataFrame(patterns)
    df_add = df_add.rename(columns = {0:"protein_conc_log", 1:"rna_conc_log"})
    df_test = pd.concat([df_add, df], axis = "index").reset_index(drop = True)
    return df_test

file_name = "chin"
max_num = 1381
df = pd.DataFrame()
df_test = pd.DataFrame()
files_path = data_dir + file_name 
files =f"{files_path}/sel_{file_name}_{min_num}_{max_num}.csv"
os.makedirs(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/souzu/", exist_ok = True)
aa_rna_list = []
group_list = []
num_list = {}
test_list = {}
df_chin_both = df_chin[df_chin.group_label.isin(df_prepro.group_label.unique())]
for group in range(df_chin_both.group_label.unique().shape[0]):
    df_test = pd.DataFrame()
    df_target_group = df_chin_both[df_chin_both.group_label == group]
    num_list[group] = []
    test_list[group] = {}
    for aa_rna in df_target_group.ex_label.unique():
        df = pd.DataFrame()
        df_test = pd.DataFrame()
        num_list[group].append(aa_rna)
        df_target = df_target_group[df_target_group.ex_label==aa_rna]
        target_x = df_target.drop(['mor_label','group_label'], axis = "columns")
        # 入力に使う濃度の範囲を拡大
        # 入力のデータ数を20 * 20 + 元　に増やした
        df_test = create_test_ex(df_target)
        
        df_test.iloc[:, 2:] = df_test.iloc[:, 2:].fillna(target_x.drop(['protein_conc_log', "rna_conc_log"], axis = "columns").iloc[0, :])
        df_test = df_test.loc[:, df_target.columns]
        df_test = df_test[df_test.mor_label.isna()]
        test_list[group][aa_rna] = df_test.copy()
        
        #増やしたデータを予測
        # モデルはLOGOの評価で、指定のグループがテストになった時のFoldを取り出して使用
        df_test["preds"] = model_chin[int(df_target.group_label.unique())].predict(df_test.drop(['mor_label', 'group_label', 'aa_rna', 'db', 'ex_label'], axis = "columns").values)
        
        pred_protein_1 = df_test[(df_test.preds == 1)].protein_conc_log
        pred_rna_1 = df_test[(df_test.preds == 1)].rna_conc_log
        pred_protein_0 = df_test[(df_test.preds == 0)].protein_conc_log
        pred_rna_0 = df_test[(df_test.preds == 0) ].rna_conc_log
        act_protein_1 = df_target[(df_target.mor_label == 1)].protein_conc_log
        act_rna_1 = df_target[(df_target.mor_label == 1)].rna_conc_log
        act_protein_0 = df_target[(df_target.mor_label == 0)].protein_conc_log
        act_rna_0 = df_target[(df_target.mor_label == 0)].rna_conc_log
        
        fig= plt.figure(figsize = (10, 10))
        plt.scatter(x = pred_protein_1, y = pred_rna_1,  c = "dodgerblue",alpha = 0.5, label = "Preds = Liquid")
        plt.scatter(x = pred_protein_0, y = pred_rna_0, c = "orange", alpha = 0.5, label = "Preds = Solute")

        plt.scatter(x = act_protein_1, y = act_rna_1, c = "mediumblue", alpha = 1, marker = "D", label = "Actual = Liquid")
        plt.scatter(x = act_protein_0, y = act_rna_0,  c = "firebrick", alpha = 1, marker = "D", label = "Actual = Solute")
        plt.xlabel("Log [Protein] (μM)", fontsize = 32, labelpad = 15)
        plt.ylabel("Log [RNA] (μM)", fontsize = 32, labelpad = 15)
#         plt.legend(fontsize = 28,  loc='lower right', bbox_to_anchor=(1.65, 0))
        plt.title(f"RNAPSEC (with SMOTE) \n Group = {group}, EX = {aa_rna}", fontsize = 36, pad = 16)
        plt.grid()
        plt.xticks(fontsize = 28)
        plt.yticks(fontsize = 28)
        fig.savefig(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/souzu/souzu_group{group}_{aa_rna}.png")


# prepro
# protein_conc_logとrna_conc_logの範囲をchinと揃える
file_name = "prepro"
max_num = 223
files_path = data_dir + file_name 
os.makedirs(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/souzu/", exist_ok = True)
for group in num_list.keys():
    if (df_prepro.group_label == group).sum() > 0:
        df_target_group = df_prepro[df_prepro.group_label == group]
#         print(1, df_target_group.ex_label.unique().shape)
        for num, aa_rna in enumerate(num_list[group]):
            df_target = df_target_group[df_target_group.ex_label==aa_rna]
#             print(2, num)
            if df_target.shape[0] > 0:
#                 print(3, group, num)
                target_x = df_target.drop(['mor_label','group_label'], axis = "columns")
                #増やしたデータを予測
                # モデルはLOGOの評価で、指定のグループがテストになった時のFoldを取り出して使用
                df_test = test_list[group][aa_rna]
                df_test["preds"] = model_prepro[num].predict(df_test.drop(['mor_label', 'group_label', 'aa_rna', 'db', 'ex_label'], axis = "columns").values)

                pred_protein_1 = df_test[(df_test.preds == 1)].protein_conc_log
                pred_rna_1 = df_test[(df_test.preds == 1)].rna_conc_log
                pred_protein_0 = df_test[(df_test.preds == 0)].protein_conc_log
                pred_rna_0 = df_test[(df_test.preds == 0) ].rna_conc_log
                act_protein_1 = df_target[(df_target.mor_label == 1)].protein_conc_log
                act_rna_1 = df_target[(df_target.mor_label == 1)].rna_conc_log
                act_protein_0 = df_target[(df_target.mor_label == 0)].protein_conc_log
                act_rna_0 = df_target[(df_target.mor_label == 0)].rna_conc_log

                fig= plt.figure(figsize = (10, 10))
                plt.scatter(x = pred_protein_1, y = pred_rna_1,  c = "dodgerblue",alpha = 0.5, label = "Preds = Liquid")
                plt.scatter(x = pred_protein_0, y = pred_rna_0, c = "orange", alpha = 0.5, label = "Preds = Solute")

                plt.scatter(x = act_protein_1, y = act_rna_1, c = "mediumblue", alpha = 1, marker = "D", label = "Actual = Liquid")
                plt.scatter(x = act_protein_0, y = act_rna_0,  c = "firebrick", alpha = 1, marker = "D", label = "Actual = Solute")
                plt.xlabel("Log [Protein] (μM)]", fontsize = 32, labelpad = 15)
                plt.ylabel("Log [RNA] (μM)]", fontsize = 32, labelpad = 15)
#                 plt.legend(fontsize = 28,  loc='lower right', bbox_to_anchor=(1.65, 0))
                plt.title(f"RNAPhaSep (with SMOTE) \n Group = {group}, EX = {aa_rna}", fontsize = 36, pad = 16)
                plt.grid()
                plt.xticks(fontsize = 28)
                plt.yticks(fontsize = 28)
                fig.savefig(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/souzu/souzu_group{group}_{aa_rna}.png")