#%%
from sklearn import preprocessing
import numpy as np
import pandas as pd

"input"
def prepare_dataset(df):
    df = df[(df.morphology_add != "gel")]
    X = x_use_col(df)
    df["mor_label"] =  labeling_mor_class(df)
    y = df.mor_label
    X, y = X.reset_index(drop = True), y.reset_index(drop = True)
    return df, X, y
def labeling_mor_class(df, ):
    classes_list = ['solute', 'liquid', 'solid',]
    le = preprocessing.LabelEncoder()
    le.fit(classes_list)
    labels_id = le.transform(df.morphology_add.values)
    labels_id
    df["mor_label"] = np.nan
    df["mor_label"][df.morphology_add == "liquid"] = 1
    df["mor_label"][df.morphology_add == "solute"] = 0
    df["mor_label"][df.morphology_add == "solid"] = 2
    df["mor_label"] = df["mor_label"][df["mor_label"].notna()].astype("int")
    return df["mor_label"]

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
    labels_id = le.fit_transform(df[label])
    df["group_label"] = labels_id 
    group_label = df["group_label"]
    return df, group_label


"ML用"
df_rna = pd.read_csv("./preprocessing_results/rna_mf_results.csv").reset_index(drop = True)
df_b = pd.read_csv("./preprocessing_results/rnap_preprocessing_bf.csv").reset_index(drop = True)
df_e = pd.read_csv("./preprocessing_results/rnap_preprocessing_is.csv").reset_index(drop = True)
df_ex1 = pd.read_csv("./preprocessing_results/rnap_preprocessing_1.csv")
assert df_rna.shape[0] == df_b.shape[0] == df_e.shape[0], "shape of input files did not matched"

df_e.rnapsec_idx


df = pd.concat([ df_e, df_b, df_rna], axis = 1)
df = df[df.pmidlink.notna()]
"saltの欠損をnp.nanで統一"
df = df.rename(columns = {"salt_unit/salt_name": "salt_unit"})
df.salt_conc = df.salt_conc.replace({"-":np.nan}, )
df.salt_conc = df.salt_conc.replace({None:np.nan})

# df.to_csv("./preprocessing_results/ours_data.csv", index = False)
# df = pd.read_csv("./preprocessing_results/ours_data.csv")
df["rnapsec_idx"] = df_ex1.rnapsec_idx
df.aa = df.aa.str.replace(">spQ99496RING2_HUMAN E3 ubiquitin-protein ligase RING2 OS=Homo sapiens OX=9606 GN=RNF2 PE=1 SV=1", "")
df_1, X, y = prepare_dataset(df)


##実験条件, flexibilityに欠損値を含むデータを削除 (flexibility-> タンパク質配列長が9以下のアミノ酸は計算できない)
def drop_ex_na(df):
    print("before", df.shape)
    df = df[df['rna_conc_log'].notna()]
    print("after rna conc.", df.shape)
    df = df[df['protein_conc_log'].notna()]
    print("after protein conc.", df.shape)
    df = df[df['pH'].notna()]
    print("after pH", df.shape)
    df = df[df['ionic_strength'].notna()]
    print("after ionic strength", df.shape)
    df = df[df['temp'].notna()] #->notnaが270件しかない
    print("after temp.", df.shape)
    df = df[df["protein_average"].notna()] #計算できる最低の配列長9以下の配列を除去
    print("after protein length", df.shape)
    # df= df[df.other_molecule_unit.isna()] #他の分子の濃度が変数になっている実験を除外
    print("after", df.shape)
    return df

df_2, label = labeling(df_1, label="pmidlink")
df_2 = drop_ex_na(df_2)

df_3 =  df[df.index.isin(df_2.index)]
df_3["pmid_label"] = df_2.group_label

df["pmid_label"] = df_2.group_label #group_labelとリンクを対応づけしたカラム
df["mor_label"] = df_2.mor_label
import os
os.makedirs("../data/", exist_ok=True)
df.to_csv("../data/ml_rnapsec_all_col.csv", index=True, index_label="rnapasec_all_col_idx")

#入力用csvに吐き出す
X = x_use_col(df_3)
y = df_2.mor_label
df_ml = pd.concat([X, y, df_2.group_label], axis = "columns").drop_duplicates()
# assert df_ml.tail(1).index == df.tail(1).index #アミノ酸配列のconcatミス
df_ml["aa"] = df.aa
df_ml["protein_sequence"] = df.protein_sequence
df_ml["rna_sequence"] = df.rna_sequence
df_ml["protein_name"] = df.rnaphasep_protein_name
df_ml.to_csv("../data/ml_rnapsec.csv", index=True, index_label="rnapsec_all_col_idx")

assert df_ml[df_ml.mor_label.isna()].shape[0] == 0, "dataframe included nan"
assert df_ml[~df_ml.duplicated()].shape[0] == df_ml.shape[0], "dataframe included duplicates"

# %%
df_ml.mor_label.value_counts()
# %%
