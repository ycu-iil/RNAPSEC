import numpy as np
import pandas as pd
import yaml
import glob
import os
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
os.makedirs("./honbun_result/", exist_ok=True)

with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
val_fold_strategy = args["val_fold_strategy"]
n_splits = args["n_splits"]
random_state = args["random_state"]

max_num = args["max_num"]
min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
val_fold = args["val_fold"]
data_dir = args["data_dir"]
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])
mor_class = np.array(args["mor_class"])
data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}

file_name_list = ["prepro", "chin", "chin_isin_prepro"]
score_list = []
file_list = []
path_list = []
for file_name in file_name_list:
    max_num_list = [50, 80, 223, 1381, 1108]
    fold_strategy_list = ["LeaveOneGroupOut", "GroupKFold", "StratifiedKFold",]
    for max_num in max_num_list:
        for fold_strategy in fold_strategy_list:
            for val_fold_strategy in fold_strategy_list:
    #             print(max_num, fold_strategy, val_fold_strategy)
                path = glob.glob(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/score.csv")
                if len(path) > 0:
                    print(pd.read_csv(path[0]))
                    file_list.append(pd.read_csv(path[0]).rename(index =
                                                                 {0:f"{file_name}/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}"}))
    score_list.append(pd.concat(file_list, axis = "index"))

df_score = pd.concat(score_list, axis = "index").reset_index().rename(columns = {"index": "idx"})
df_ds = df_score.idx.str.split("/", expand = True ).loc[:, [0, 1]]
df_cv = df_score.idx.str.split("/", expand = True ).loc[:, [2]]
df_cv = df_cv[2].str.split("_", expand = True).rename(columns = {1: "test", 3: "val"})
df_total_score = pd.concat([df_ds, df_cv.loc[:, ["test", "val"]], df_score], axis = "columns").sort_values(by = ["test", "val"])
df_total_score.to_csv("./honbun_result/all_results_smote_dir.csv")