import argparse
import numpy as np
import pandas as pd
import yaml
import glob
import pickle
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
val_fold_strategy = "StratifiedKFold"
n_splits = args["n_splits"]
random_state = args["random_state"]

mor_class = np.array(args["mor_class"])
data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}


min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
val_fold = 5
data_dir = args["data_dir"]

fig = plt.figure(figsize=(10, 10))
file_list = []

file_name_list = [ "prepro","chin", ]
db_name_list = ["RNAPhaSep (Curated)", "RNAPSEC (Curated)", ]
colors = ["blue", "darkred", ]
max_num_list = [ 223, 1381,]
for color, file_name, db_name, max_num in zip(colors, file_name_list, db_name_list, max_num_list,):
    
    files = data_dir + file_name + '.csv'
    df_chin = pd.read_csv(files, index_col=False)
    mor = [["solute", "liquid", "solid"][i] for i in mor_class]
    df_chin = df_chin[df_chin[target_label].isin(mor_class)]

    file_list = sorted(glob.glob(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/pred_cutdata_val_{min_num}_{max_num}_fold*.csv"))

    def calc_roc(file_list):
        fpr_list = []
        tpr_list = []
        auc_list= []
        for i, filename in enumerate(file_list):
            test = pd.read_csv(filename, )
            y_pred=test['liquid']
            y_test=test['actual']
            #AUCを求める
            fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
            auc = metrics.auc(fpr,tpr)
            # plt.plot(fpr, tpr,  color = color,alpha = 0.3, lw = 0.9)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(auc)
        return fpr_list, tpr_list, auc_list
    fpr_list, tpr_list, auc_list = calc_roc(file_list)
    for fpr, tpr, auc in zip(fpr_list, tpr_list, auc_list):
        plt.plot(fpr, tpr,  color = color,alpha = 0.3, lw = 0.9, c = color)

    #平均のAUCを求める
    test= pd.read_csv(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/total_result_val_{min_num}_{max_num}.csv")

    y_pred_ave=test["liquid"]
    y_test=test['actual']
    fpr_ave, tpr_ave, thresholds_ave = metrics.roc_curve(y_test,y_pred_ave)
    auc_ave = metrics.auc(fpr_ave,tpr_ave)

    #平均のROC曲線を描く
    plt.plot(fpr_ave, tpr_ave, label=f'{db_name}\nROC-AUC = %.4f'%auc_ave, color = color, alpha = 1.0,  lw = 1.6)
plt.legend(fontsize = 20, loc = "lower right")
plt.grid()
plt.xlabel("False Positive rate", fontsize = 24)
plt.ylabel("True Positive rate", fontsize = 24)
plt.title(f"ROC ({fold_strategy})", fontsize = 32, pad = 15)
plt.show()
fig.savefig(f"./honbun_result/fig11ab_{fold_strategy}.png")