
#%%
#学習データ(val, train)にsmoteを適応
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut, cross_val_score, cross_validate
import numpy as np
import pandas as pd
import yaml
import pickle
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

#trainだけsmoteかける。validationとtestはかけない
#交差検証とROC曲線、valデータ付き (分割方法はconfig_.yamlで指定）、パラメーター調整なし

with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
val_fold_strategy = args["val_fold_strategy"]


file_name = args["file_name"]
mor_class = np.array(args["mor_class"])

# max_num = args["max_num"]
min_num = args["min_num"]

val_fold = args["val_fold"]



d = 20230719
files =f"../data/{file_name}.csv"
print(files)
df_rnapsec = pd.read_csv(files, index_col=False) 
#%%
#%%
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_rnapsec = df_rnapsec[df_rnapsec[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)
max_num= df_rnapsec.shape[0]
#X, y, groups
if ignore_features == False:  #省くカラムがない場合   
    X = df_rnapsec.drop([target_label, group_label, "aa_rna_label", "protein_sequence", "rna_sequence","protein_name", "rnapsec_all_col_idx" ], axis = "columns").reset_index(drop = True).values
    print(X.shape)
else:
    print(ignore_features)
    X = df_rnapsec.drop([target_label, group_label], axis = "columns").reset_index(drop = True)
    X = X.drop(ignore_features, axis = "columns").reset_index(drop = True).values
y = df_rnapsec[target_label].values
groups = df_rnapsec[group_label].values
print(df_rnapsec[group_label].unique().shape)
print(df_rnapsec.columns)
#%%
def create_test_ex(df):
    protein_conc = np.linspace(df.protein_conc_log.min()-0.5, df.protein_conc_log.max()+0.5, 20)
    protein_conc.shape
    rna_conc = np.linspace(df.rna_conc_log.min()-0.5, df.rna_conc_log.max()+0.5, 20)
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



# max_num = args["max_num"]
df = pd.DataFrame()
df_test = pd.DataFrame()
files =f"../data/{file_name}.csv"

# for model in ["AdaBoostClassifier", "LGBMClassifier"]:
model = "LGBMClassifier"

std = "True"
random_state = "00"
result_path = f"./result_logocv/{model}"
os.makedirs(f"{result_path}/souzu_{d}/", exist_ok = True)

aa_rna_list = []
group_list = []
num_list = {}
test_list = {}

with open(f"{result_path}/model_list_{file_name}.pickle", 'rb') as web:
    model_rnapsec = pickle.load(web)
for group in range(df_rnapsec[group_label].unique().shape[0]):
    df_target_group = pd.DataFrame()
    df_target_group = df_rnapsec[df_rnapsec[group_label] == group].sort_values(by = "aa_rna_label")
    num_list[group] = []
    test_list[group] = {}
    result_list = []
    for aa_rna in df_target_group.aa_rna_label.unique():
        i= group
        print("model_idx = ", i)
        print("aa: ", group, )
        print("ex: ", aa_rna)
        df = pd.DataFrame()
        df_test = pd.DataFrame()
        num_list[group].append(aa_rna)
        df_target = df_target_group[df_target_group.aa_rna_label==aa_rna]
        target_x = df_target.drop(['mor_label',group_label, "rnapsec_all_col_idx"], axis = "columns")
        # 入力に使う濃度の範囲を拡大
        # 入力のデータ数を20 * 20 + 元　に増やした
        df_test = create_test_ex(df_target)
        
        df_test.iloc[:, 2:] = df_test.iloc[:, 2:].fillna(target_x.drop(['protein_conc_log', "rna_conc_log"], axis = "columns").iloc[0, :])
        df_test = df_test.loc[:, df_target.columns]
        df_test = df_test[df_test.mor_label.isna()]
        test_list[group][aa_rna] = df_test.copy()
        
        #増やしたデータを予測
        # モデルはLOGOの評価で、指定のグループがテストになった時のFoldを取り出して使用
        X_souzu = df_test.drop([target_label, group_label, "aa_rna_label", "protein_sequence", "rna_sequence","protein_name", "rnapsec_all_col_idx" ], axis = "columns").values
        print("X_souzu.shape: ", X_souzu.shape)
        print("test group: ", df_test.aa_rna_label.unique())
        df_test["preds"] = model_rnapsec[i].predict(X_souzu)
        df_test["proba"] = model_rnapsec[i].predict_proba(X_souzu)[:, 1]
        
        result_list.append(df_test)#.drop("aa_rna_label", axis="columns")
        test_list[group][aa_rna] = df_test.copy()
        pred_protein_1 = df_test[(df_test.preds == 1)].protein_conc_log
        pred_rna_1 = df_test[(df_test.preds == 1)].rna_conc_log
        pred_protein_0 = df_test[(df_test.preds == 0)].protein_conc_log
        pred_rna_0 = df_test[(df_test.preds == 0) ].rna_conc_log
        act_protein_1 = df_target[(df_target.mor_label == 1)].protein_conc_log
        act_rna_1 = df_target[(df_target.mor_label == 1)].rna_conc_log
        act_protein_0 = df_target[(df_target.mor_label == 0)].protein_conc_log
        act_rna_0 = df_target[(df_target.mor_label == 0)].rna_conc_log
        
        fig, ax = plt.subplots(figsize = (10, 10))
        # plot_setting()
        ax.scatter(x = pred_protein_1, y = pred_rna_1, c = "lightskyblue",alpha = 1, label = "Prediction: Liquid", s = 620, marker = "s") # c = "dodgerblue"
        ax.scatter(x = pred_protein_0, y = pred_rna_0, c = "navajowhite", alpha = 1, label = "Prediction: Solute", s =620, marker = "s")
        ax.scatter(x = act_protein_1, y = act_rna_1, c = "mediumblue", alpha = 1, marker = "D", label = "Experiment: Liquid", s = 200)
        ax.scatter(x = act_protein_0, y = act_rna_0,  c = "firebrick",marker = "D", alpha = 1,  label = "Experiment: Solute", s =200)
        ax.set_xlabel("Protein conc. (log μM)", fontsize = 26, labelpad = 15)
        ax.set_ylabel("RNA conc. (log μM)", fontsize = 26, labelpad = 15)
        ax.xaxis.set_major_locator(ticker.LinearLocator(6))
        ax.xaxis.set_tick_params(direction='out', labelsize=20, width=3, pad=10)
        ax.yaxis.set_tick_params(direction='out', labelsize = 20, width=3, pad=10)
        ax.legend(fontsize = 28,  loc='lower right', bbox_to_anchor=(1.8, 0))
        ax.set_title(f"{model}, {group_label},\n Group = {group}, EX = {aa_rna}, {fold_strategy}", fontsize = 26, pad = 24)
        fig.savefig(f"{result_path}/souzu_{d}/souzu_group{group}_{aa_rna}.png", bbox_inches='tight', transparent=False)
        plt.close()
        aa_rna_list.append("a") 
    os.makedirs(f"{result_path}/souzu_result_{d}/", exist_ok=True)
    if len(result_list) >0:
        # print(group)
        # print(len(result_list))
        pd.concat(result_list, axis="index").to_csv(f"{result_path}/souzu_result_{d}/souzu_result_{group}.csv", index=False) #僧都書くときに使ったデータを保存
