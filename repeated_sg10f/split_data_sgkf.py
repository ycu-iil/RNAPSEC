import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
import os
def main():
    with open("./config_.yaml",'r')as f:
        args = yaml.safe_load(f)
    target_label = args["target_label"]
    group_label = args["group_label"]
    ignore_features = args["ignore_features"]
    n_splits = args["n_splits"]
    file_name = args["file_name"]
    mor_class = np.array(args["mor_class"])
    files =f"../data/{file_name}.csv"
    df_rnapsec = pd.read_csv(files, index_col=False) 
    df_rnapsec = df_rnapsec[df_rnapsec[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True)

    if ignore_features == False:  
        X = df_rnapsec.drop([target_label, group_label, "rnapsec_all_col_idx",
        'protein_sequence', 'rna_sequence', 'protein_name', 
        'aa_rna_label'], axis = "columns").reset_index(drop = True).values
    else:
        print(ignore_features)
        X = df_rnapsec.drop([target_label, group_label], axis = "columns").reset_index(drop = True)
        X = X.drop(ignore_features, axis = "columns").reset_index(drop = True).values
    y = df_rnapsec[target_label].values
    groups = df_rnapsec[group_label].values

    for kf_state in range(10, 101, 10):
        kf = StratifiedGroupKFold(n_splits = n_splits, shuffle = True, random_state = kf_state)
        os.makedirs(f"./result_repeated_StratifiedGroupKFold/split_kf_state_{kf_state}", exist_ok=True)
        for x, (train_val_index, test_index) in enumerate(kf.split(X, y, groups = groups )):
            df_rnapsec["fold_num"] = x
            df_rnapsec.iloc[test_index, :].to_csv(f"./result_repeated_StratifiedGroupKFold/split_kf_state_{kf_state}/test_df_{group_label}_fold{x}.csv", index=False)

if __name__=="__main__":
    main()