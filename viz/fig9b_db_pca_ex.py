# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
from sklearn.preprocessing import StandardScaler

def extract_db(df_psec, ml_psec):
    mor_class = [0, 1]
    target_label = "mor_label"
    group_label = "group_label"
    ignore_features = False
    mor = [["solute", "liquid", "solid"][i] for i in mor_class]
    ml_psec = ml_psec[ml_psec[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
#     ml_psec= ml_psec.dropna()
    #X, y, groups
    if ignore_features == False:  #省くカラムがない場合   
        X = ml_psec.drop([target_label, group_label], axis = "columns").reset_index(drop = True).values
    else:
        X = ml_psec.drop([target_label, group_label, ignore_features], axis = "columns").reset_index(drop = True).values
    y = ml_psec[target_label].values
    groups = ml_psec[group_label].values
    return X, y, groups, ml_psec
def standard(X, ml_psec):
    #標準化
    std_sc = StandardScaler()
    std_sc.fit(X)
    X_std = std_sc.transform(X)
    df_X_std = pd.DataFrame(X_std, columns = ml_psec.drop(["group_label", "mor_label"], axis = "columns").columns)
    df_ml_psec_std = pd.concat([df_X_std, ml_psec.loc[:, ["group_label", "mor_label"]].reset_index(drop = True)], axis = "columns")
    return df_ml_psec_std
    
def sep_by_feature(df_ml_psec_std):
    #特徴量の分類ごとにデータフレーム作成
    df_ml_psec_std_ex = df_ml_psec_std.loc[:, [ 'rna_conc_log',
        'protein_conc_log', 'pH', 'temp', 'ionic_strength',]]
    df_ml_psec_std_aa =  df_ml_psec_std.loc[:, df_ml_psec_std.columns[df_ml_psec_std.columns.str.contains("protein")]]
    df_ml_psec_std_rna =  df_ml_psec_std.loc[:, df_ml_psec_std.columns[df_ml_psec_std.columns.str.contains("rna")]]
    df_ml_psec_std_rna.columns = df_ml_psec_std_rna.columns.str.replace("_rna_", "_")
    groups = df_ml_psec_std["group_label"]
    return df_ml_psec_std_ex, df_ml_psec_std_aa, df_ml_psec_std_rna

df_phasep = pd.read_csv("../data/prepro_all_col.csv", index_col = False)
ml_phasep = pd.read_csv("../data/prepro.csv", index_col = False)
df_psec = pd.read_csv("../data/chin_all_col.csv", index_col = False)
ml_psec = pd.read_csv("../data/chin.csv", index_col = False)
phasep = "RNAPhaSep (Curated)"
psec = "RNAPSEC (Curated)"

os.makedirs("./pca/", exist_ok=True)
ml_phasep["ds_name"] = "prepro"
ml_psec["ds_name"] = "chin"
ml_con = pd.concat([ml_phasep, ml_psec], axis = "index")

X_con, y_con, groups_con, ml_con2 = extract_db(df_psec, ml_con)
X_con = X_con[:, :-1] #ds_nameのカラムを落とす

mor_class = [0,1]
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
ml_con = ml_con[ml_con.mor_label.isin(mor_class)] 

#標準化
std_sc = StandardScaler()
std_sc.fit(X_con)
X_std = std_sc.transform(X_con)
df_X_std = pd.DataFrame(X_std, columns = ml_con.drop(["group_label", "mor_label", "ds_name"], axis = "columns").columns)
df_con_std = pd.concat([df_X_std, ml_con.loc[:, ["group_label", "mor_label", "ds_name"]].reset_index(drop = True)], axis = "columns")


df_con_std_ex, df_con_std_aa, df_con_std_rna = sep_by_feature(df_con_std)
target_df_list = [df_con_std_ex, df_con_std_aa, df_con_std_rna]
fig_num_list = ["b", "c", "d"]
feature_list = ["ex", "protein", "rna"]
title_list = ["Experimental Conditions", "Protein features", "RNA features"]

for df_con_std_target, fig_num, feature, title in zip(target_df_list, fig_num_list, feature_list, title_list):
    X_con = df_con_std_ex.values
    df_con_std_chin = df_con_std_ex[df_con_std.ds_name == "chin"]
    X_chin = df_con_std_chin.values
    df_con_std_prepro = df_con_std_ex[df_con_std.ds_name == "prepro"]
    X_prepro = df_con_std_prepro.values
    pca = PCA()
    pca.fit(X_con)
    feature_chin = pca.transform(X_chin)
    feature_prepro = pca.transform(X_prepro)

    df_pca_chin = pd.DataFrame(feature_chin).loc[:, [0, 1]]
    df_pca_prepro = pd.DataFrame(feature_prepro).loc[:, [0, 1]]
    #両方のデータセットに入っているデータ、RNAPhaSepのみ、RNAPSECのみのデータに分ける
    df_pca_1 = pd.concat([df_pca_chin, df_pca_prepro], axis = "index")
    df_pca_chin["ds_name"] = "chin"
    df_pca_prepro["ds_name"] = "prepro"
    df_pca_chin_dup = df_pca_chin[(df_pca_chin[0].isin(df_pca_prepro[0])) & (df_pca_chin[1].isin(df_pca_prepro[1]))]
    df_pca_chin_unique = df_pca_chin[~((df_pca_chin[0].isin(df_pca_prepro[0])) & (df_pca_chin[1].isin(df_pca_prepro[1])))]
    df_pca_prepro_dup = df_pca_prepro[(df_pca_prepro[0].isin(df_pca_chin[0])) & (df_pca_prepro[1].isin(df_pca_chin[1]))]
    df_pca_prepro_unique = df_pca_prepro[~((df_pca_prepro[0].isin(df_pca_chin[0])) & (df_pca_prepro[1].isin(df_pca_chin[1])))]
    # 第一主成分と第二主成分でプロットする
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(df_pca_prepro_unique.loc[:, 0], df_pca_prepro_unique.loc[:, 1], alpha=0.5, label = "RNAPhaSep (only)", c = "orange", s = 20)
    plt.scatter(df_pca_chin_unique.loc[:, 0], df_pca_chin_unique.loc[:, 1], alpha=0.5, label = "RNAPSEC (only)", c = "dodgerblue", s = 20)
    plt.scatter(df_pca_prepro_dup.loc[:, 0], df_pca_prepro_dup.loc[:, 1], alpha=1, label = "Both", c = "grey", marker = "D", s = 25)
    plt.grid()
    plt.xlabel("PC1", fontsize = 20)
    plt.ylabel("PC2", fontsize = 20)
    plt.xticks(fontsize  =16)
    plt.yticks(fontsize  =16)
    # plt.legend(fontsize = 18, loc = "upper right", bbox_to_anchor=(1.55, 1))
    plt.title(f"{title}", fontsize = 28, pad = 18)
    plt.show()
    fig.savefig(f"./pca/fig9{fig_num}_pca_{feature}.png")

