#%%
import pandas as pd
from sklearn import preprocessing

def labeling(df, label="protein_name_seq"):
    le = preprocessing.LabelEncoder()
    labels_id = le.fit_transform(df[label])
    df[f"{label}_label"] = labels_id 
    return df

ml_rnapsec = pd.read_csv("./prepro_results/preprocessing_result_ex.csv", index_col=False)
# ラベリング
ml_rnapsec_2 = labeling(ml_rnapsec, label = "aa") #アミノ酸配列のラベリング
# アミノ酸配列とRNA配列ラベリング
ml_rnapsec_2["aa_rna"] = ml_rnapsec_2.aa + ml_rnapsec_2.rna_sequence
ml_rnapsec_2 = labeling(ml_rnapsec_2, label = "aa_rna")
ml_rnapsec_2.to_csv("./prepro_results/preprocessing_result_ex_2.csv", index=False)

# %%
ml_rnapsec_2
# %%
