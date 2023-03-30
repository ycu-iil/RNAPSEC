
from sklearn import preprocessing

import scipy.sparse as sp
import numpy as np
import pandas as pd
import yaml
"input"
def prepare_dataset(df):
    
    df = df[(df.morphology_add != "gel")]
    df = df.loc[:,['protein_weight', 'protein_A', 'protein_C', 'protein_D', 'protein_E',
       'protein_F', 'protein_G', 'protein_H', 'protein_I', 'protein_K',
       'protein_L', 'protein_M', 'protein_N', 'protein_P', 'protein_Q',
       'protein_R', 'protein_S', 'protein_T', 'protein_V', 'protein_W',
       'protein_Y', 'protein_aromaticity', 'protein_instability',
       'protein_gravy', 'protein_average', 'protein_isopt', 'protein_helix',
       'protein_turn', 'protein_sheet', 
        'rna_rna_A', 'rna_rna_C', 'rna_rna_G', 'rna_rna_U', 'rna_rna_AA', 'rna_rna_AC',
        'rna_rna_AG', 'rna_rna_AU', 'rna_rna_CA', 'rna_rna_CC', 'rna_rna_CG', 'rna_rna_CU', 'rna_rna_GA',
        'rna_rna_GC', 'rna_rna_GG', 'rna_rna_GU', 'rna_rna_UA', 'rna_rna_UC', 'rna_rna_UG', 'rna_rna_UU',
        'rna_rna_AAA', 'rna_rna_AAC', 'rna_rna_AAG', 'rna_rna_AAU', 'rna_rna_ACA', 'rna_rna_ACC',
        'rna_rna_ACG', 'rna_rna_ACU', 'rna_rna_AGA', 'rna_rna_AGC', 'rna_rna_AGG', 'rna_rna_AGU',
        'rna_rna_AUA', 'rna_rna_AUC', 'rna_rna_AUG', 'rna_rna_AUU', 'rna_rna_CAA', 'rna_rna_CAC',
        'rna_rna_CAG', 'rna_rna_CAU', 'rna_rna_CCA', 'rna_rna_CCC', 'rna_rna_CCG', 'rna_rna_CCU',
        'rna_rna_CGA', 'rna_rna_CGC', 'rna_rna_CGG', 'rna_rna_CGU', 'rna_rna_CUA', 'rna_rna_CUC',
        'rna_rna_CUG', 'rna_rna_CUU', 'rna_rna_GAA', 'rna_rna_GAC', 'rna_rna_GAG', 'rna_rna_GAU',
        'rna_rna_GCA', 'rna_rna_GCC', 'rna_rna_GCG', 'rna_rna_GCU', 'rna_rna_GGA', 'rna_rna_GGC',
        'rna_rna_GGG', 'rna_rna_GGU', 'rna_rna_GUA', 'rna_rna_GUC', 'rna_rna_GUG', 'rna_rna_GUU',
        'rna_rna_UAA', 'rna_rna_UAC', 'rna_rna_UAG', 'rna_rna_UAU', 'rna_rna_UCA', 'rna_rna_UCC',
        'rna_rna_UCG', 'rna_rna_UCU', 'rna_rna_UGA', 'rna_rna_UGC', 'rna_rna_UGG', 'rna_rna_UGU',
        'rna_rna_UUA', 'rna_rna_UUC', 'rna_rna_UUG', 'rna_rna_UUU',
        'rna_fickett_score-ORF', 'rna_fickett_score-full-sequence',
       'rna_tsallis_k1', 'rna_tsallis_k2', 'rna_real_avg', 'rna_real_peak',
       'rna_binary_avg', 'rna_binary_peak', 'rna_zcurve_avg',
       'rna_zcurve_peak', 'rna_shanon_k1', 'rna_shanon_k2',
       'rna_cv_ORF_length',"rna_conc_log", "protein_conc_log", "pH", "temp", "ionic_strength",
        "morphology_add","pmidlink", "aa", "rna_sequence", "ds_name"]]
    X = x_use_col(df)
    df["mor_label"] =  labeling_mor_class(df, classes_list = ["solute", "liquid", "solid"])
    y = df.mor_label
    X, y = X.reset_index(drop = True), y.reset_index(drop = True)
    return df, X, y
def labeling_mor_class(df, classes_list = ['solute', 'liquid', 'solid',]):
    le = preprocessing.LabelEncoder()
    le.classes_ = classes_list
    labels_id = le.transform(df.morphology_add)
    labels_id
    return labels_id

def x_use_col(df):
    X = df.loc[:, ['protein_weight', 'protein_A', 'protein_C', 'protein_D', 'protein_E',
       'protein_F', 'protein_G', 'protein_H', 'protein_I', 'protein_K',
       'protein_L', 'protein_M', 'protein_N', 'protein_P', 'protein_Q',
       'protein_R', 'protein_S', 'protein_T', 'protein_V', 'protein_W',
       'protein_Y', 'protein_aromaticity', 'protein_instability',
       'protein_gravy', 'protein_average', 'protein_isopt', 'protein_helix',
       'protein_turn', 'protein_sheet',
        'rna_rna_A', 'rna_rna_C', 'rna_rna_G', 'rna_rna_U', 'rna_rna_AA', 'rna_rna_AC',
        'rna_rna_AG', 'rna_rna_AU', 'rna_rna_CA', 'rna_rna_CC', 'rna_rna_CG', 'rna_rna_CU', 'rna_rna_GA',
        'rna_rna_GC', 'rna_rna_GG', 'rna_rna_GU', 'rna_rna_UA', 'rna_rna_UC', 'rna_rna_UG', 'rna_rna_UU',
        'rna_rna_AAA', 'rna_rna_AAC', 'rna_rna_AAG', 'rna_rna_AAU', 'rna_rna_ACA', 'rna_rna_ACC',
        'rna_rna_ACG', 'rna_rna_ACU', 'rna_rna_AGA', 'rna_rna_AGC', 'rna_rna_AGG', 'rna_rna_AGU',
        'rna_rna_AUA', 'rna_rna_AUC', 'rna_rna_AUG', 'rna_rna_AUU', 'rna_rna_CAA', 'rna_rna_CAC',
        'rna_rna_CAG', 'rna_rna_CAU', 'rna_rna_CCA', 'rna_rna_CCC', 'rna_rna_CCG', 'rna_rna_CCU',
        'rna_rna_CGA', 'rna_rna_CGC', 'rna_rna_CGG', 'rna_rna_CGU', 'rna_rna_CUA', 'rna_rna_CUC',
        'rna_rna_CUG', 'rna_rna_CUU', 'rna_rna_GAA', 'rna_rna_GAC', 'rna_rna_GAG', 'rna_rna_GAU',
        'rna_rna_GCA', 'rna_rna_GCC', 'rna_rna_GCG', 'rna_rna_GCU', 'rna_rna_GGA', 'rna_rna_GGC',
        'rna_rna_GGG', 'rna_rna_GGU', 'rna_rna_GUA', 'rna_rna_GUC', 'rna_rna_GUG', 'rna_rna_GUU',
        'rna_rna_UAA', 'rna_rna_UAC', 'rna_rna_UAG', 'rna_rna_UAU', 'rna_rna_UCA', 'rna_rna_UCC',
        'rna_rna_UCG', 'rna_rna_UCU', 'rna_rna_UGA', 'rna_rna_UGC', 'rna_rna_UGG', 'rna_rna_UGU',
        'rna_rna_UUA', 'rna_rna_UUC', 'rna_rna_UUG', 'rna_rna_UUU',
        'rna_fickett_score-ORF', 'rna_fickett_score-full-sequence',
       'rna_tsallis_k1', 'rna_tsallis_k2', 'rna_real_avg', 'rna_real_peak',
       'rna_binary_avg', 'rna_binary_peak', 'rna_zcurve_avg',
       'rna_zcurve_peak', 'rna_shanon_k1', 'rna_shanon_k2',
       'rna_cv_ORF_length',"rna_conc_log", "protein_conc_log", "pH", "temp", "ionic_strength",]]
    return X
def labeling(df, label="pmidlink"):
    le = preprocessing.LabelEncoder()
    labels = df[label].unique()
    labels_id = le.fit_transform(df[label])
    df["group_label"] = labels_id 
    group_label = df["group_label"]
    return df, group_label


"ML用"
df_chin_r = pd.read_csv("./preprocessing_results/rna_mf_results.csv")
df_chin_b = pd.read_csv("./preprocessing_results/rnap_preprocessing_bf.csv")
df_chin_e = pd.read_csv("./preprocessing_results/rnap_preprocessing_is.csv")
# df_chin_rna = df_chin_r.iloc[1:]
df_chin_rna = df_chin_r.reset_index(drop = True)

df_chin = pd.concat([ df_chin_e, df_chin_b, df_chin_rna], axis = 1)
df_chin = df_chin[df_chin.pmidlink.notna()]
"saltの欠損をnp.nanで統一"
df_chin = df_chin.rename(columns = {"salt_unit/salt_name": "salt_unit"})
df_chin.salt_conc = df_chin.salt_conc.replace({"-":np.nan}, )
df_chin.salt_conc = df_chin.salt_conc.replace({None:np.nan})

# df_chin.to_csv("./preprocessing_results/ours_data.csv", index = False)
# df_chin = pd.read_csv("./preprocessing_results/ours_data.csv")
df_chin["ds_name"] = "chin"
df_chin.aa = df_chin.aa.str.replace(">spQ99496RING2_HUMAN E3 ubiquitin-protein ligase RING2 OS=Homo sapiens OX=9606 GN=RNF2 PE=1 SV=1", "")
"prepro"
df_prepro_ex =  pd.read_csv("../preprocessing_rnaphasep/prepro_results/rnap_calc_is_6.csv").drop("protein_weight", axis = "columns")
# .loc[:, [ 'pmidlink', 'aa', "rna_sequence",
#         'rna_weight', 'rna_conc_uM', 'protein_conc_uM',"ini_idx", 
#        'rna_conc_log', 'protein_conc_log', 'temp', 'ph', 'morphology_add']]
df_prepro_bf = pd.read_csv("../preprocessing_rnaphasep/prepro_results/rnap_prepro_bf_4.csv")
df_prepro_rna = pd.read_csv("../preprocessing_rnaphasep/prepro_results/rna_mf_results.csv")
# df_prepro_rna=df_prepro_rna.rename(columns = {'rna_rna_nameseq': "ini_idx"})
df_prepro_ex = df_prepro_ex.rename(columns = { "ph": "pH"})
# df_prepro_bf = df_prepro_bf.loc[:, df_prepro_bf.columns[~df_prepro_bf.columns.isin(df_prepro_ex.columns)]] #重複しているから目があるので、落とす
df_prepro_0 = pd.concat([df_prepro_ex, df_prepro_bf, df_prepro_rna], axis = "columns")
df_prepro_0["ds_name"] = "prepro"
df_prepro_0 = df_prepro_0[(df_prepro_0.rna_conc_uM > 0) & (df_prepro_0.protein_conc_uM > 0)] 
df_prepro_0 = df_prepro_0[(df_prepro_0.morphology_add != "gel") & (df_prepro_0.morphology_add.notna())] 

df_chin_1, X_chin, y_chin = prepare_dataset(df_chin)
df_prepro_1, X_prepro, y_prepro = prepare_dataset(df_prepro_0)
"論文IDでラベリング -> 共通論文には絞っていない"
df_label = pd.concat([df_chin_1, df_prepro_1], axis = "index")
df_label, label = labeling(df_label, label="pmidlink")
##実験条件, flexibilityに欠損値を含むデータを削除 (flexibility-> タンパク質配列長が9以下のアミノ酸は計算できない)
def drop_ex_na(df):
    print("before", df.shape)
    df = df[df['rna_conc_log'].notna()]
    df = df[df['protein_conc_log'].notna()]
    df = df[df['pH'].notna()]
    df = df[df['ionic_strength'].notna()]
    df = df[df['temp'].notna()]
    df = df[df["protein_average"].notna()] #計算できる最低の配列長9以下の配列を除去
    print("after", df.shape)
    return df

df_chin_2 = df_label[df_label.ds_name == "chin"]
df_prepro_2 = df_label[df_label.ds_name == "prepro"]
df_chin_2 = drop_ex_na(df_chin_2)
df_prepro_2 = drop_ex_na(df_prepro_2)

df_chin_3 =  df_chin[df_chin.index.isin(df_chin_2.index)]
df_chin_3["pmid_label"] = df_chin_2.group_label

df_chin["pmid_label"] = df_chin_2.group_label #group_labelとリンクを対応づけしたカラム
df_chin["mor_label"] = df_chin_2.mor_label

df_prepro_0["pmid_label"] = df_prepro_2.group_label
df_prepro_0["mor_label"] = df_prepro_2.mor_label




df_chin.to_csv("../data/chin_all_col.csv", index = False)
df_chin.to_csv("./preprocessing_results/ours_data.csv", index = False) #全カラムの結合版


# df_prepro.to_csv("../data/prepro_all_col.csv")

#入力用csvに吐き出す
X_chin = x_use_col(df_chin_3)
y_chin = df_chin_2.mor_label

X_prepro = x_use_col(df_prepro_2)
y_prepro = df_prepro_2.mor_label
ml_chin = pd.concat([X_chin, y_chin, df_chin_2.group_label], axis = "columns")
# ml_chin.dropna().to_csv("../data/chin.csv", index = False)
ml_chin.to_csv("../data/chin.csv", index = False)
ml_prepro = pd.concat([X_prepro, y_prepro, df_prepro_2.group_label, ], axis = "columns")
# ml_prepro.dropna().to_csv("../data/prepro.csv", index = False)
ml_prepro.to_csv("../data/prepro.csv", index = False)

#mlデータカラムとrnaphasepオリジナルのカラムを結合させる
df_ours = pd.read_csv("./preprocessing_results/ours_data.csv")
df_ours = df_ours.rename(columns = {"index": "ini_idx"})
df_ours.ini_idx = df_ours.ini_idx.astype("int")

rnaphasep = pd.read_excel("./data/All data.xlsx", engine = "openpyxl") #RNAPhaSep(HP)からダウンロードしたファイルの読み込み
rnaphasep = rnaphasep.drop(rnaphasep.columns[rnaphasep.columns.str.contains("Unnamed")], axis = "columns")
rnaphasep = rnaphasep.rename(columns = {"index": "ini_idx"}) #chinのインデックスと名前を揃える
df_rnaphasep = rnaphasep[rnaphasep.ini_idx.isin(df_ours.ini_idx)] #chinとconcatするRNAPhasep列に限定したデータフレームを作成
#chinとRNAPhaSepを結合
df_ours_all_col = pd.merge(df_ours,df_rnaphasep.loc[:,["ini_idx", 'pmid', 'rna_id', 'rna_id_interface', 'rnas', 'rna_length',
       'rna_classification', 'source', 'protein_name', 'protein_region',
       'protein_modification', 'protein_sequence_length', 'Uniprot ID', 'mark',
       'link', 'IDR', 'low complexity domain', 'other_requirement',
       'detection_method', 'description']], on = "ini_idx")
df_prepro = pd.concat([df_prepro_0.loc[:, df_prepro_0.columns[~df_prepro_0.columns.isin(ml_prepro.columns)]], ml_prepro], axis = "columns")
df_prepro.to_csv("../data/prepro_all_col.csv")
#chin overlap prepro
with open("../main_cutdata/config_.yaml",'r')as f:
    args = yaml.safe_load(f)
target_label = args["target_label"]
mor_class = np.array(args["mor_class"])
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
ml_chin1 = ml_chin[ml_chin[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
ml_prepro1 = ml_prepro[ml_prepro[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
df_chin_isin_prepro = ml_chin1[ml_chin1.group_label.isin(ml_prepro1.group_label)]
df_chin_notin_prepro = ml_chin1[~ml_chin1.group_label.isin(ml_prepro1.group_label)]

df_chin_isin_prepro.to_csv("../data/chin_isin_prepro.csv", index=False)
df_chin_notin_prepro.to_csv("../data/chin_notin_prepro.csv", index=False)