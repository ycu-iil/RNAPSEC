#%%
import pandas as pd
from sklearn import preprocessing

def labeling(df, label="protein_name_seq"):
    le = preprocessing.LabelEncoder()
    labels_id = le.fit_transform(df[label])
    df[f"{label}_label"] = labels_id 
    group_label = df[f"{label}_label"]
    return df
def main():
    ml_rnapsec = pd.read_csv("../data/ml_rnapsec.csv", index_col=False).dropna().reset_index(drop=True)
    print(ml_rnapsec.mor_label.value_counts())
    # ラベリング
    ml_rnapsec_2 = labeling(ml_rnapsec, label = "aa") #アミノ酸配列のラベリング
    # アミノ酸配列とRNA配列ラベリング
    ml_rnapsec_2["aa_rna"] = ml_rnapsec_2.aa + ml_rnapsec_2.rna_sequence
    ml_rnapsec_2 = labeling(ml_rnapsec_2, label = "aa_rna")
    ml_rnapsec_2.drop(['group_label', 'aa', 'aa_rna',], axis="columns").to_csv("../data/ml_rnapsec_aa.csv", index=False)
    assert ml_rnapsec_2.aa.unique().shape[0] == ml_rnapsec_2.aa_label.unique().shape[0], "AA seq labeling error: the number of unique aa seq did not match to the number of unique aa label. " #配列種類と配列のラベル種類数があってるか。(ラベリングミスがないかを確)
    print(ml_rnapsec_2.shape)
    ml_protein_name = labeling(ml_rnapsec, label = "protein_name")
    ml_protein_name.drop(['group_label', 'aa','aa_label', 'aa_rna',], axis = "columns").to_csv("../data/ml_rnapsec_protein_name.csv", index=False)
    return

if __name__ == "__main__":
    main()

# %%
