import argparse
import numpy as np
import pandas as pd
import yaml
import glob
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut, cross_val_score, cross_validate
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)
os.makedirs("./honbun_result/", exist_ok=True)
target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
val_fold_strategy = args["val_fold_strategy"]
n_splits = args["n_splits"]
random_state = args["random_state"]

mor_class = np.array(args["mor_class"])
data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}


min_num = args["min_num"]
max_num = args["max_num"]
# max_num = 1381
# max_num = 223
early_stopping_num = args["early_stopping"]
val_fold = 5
data_dir = args["data_dir"]
file_name = args["file_name"]
# file_name = "chin"
# file_name = "prepro"
#図のタイトル設定
if file_name == "prepro":
    title = f"RNAPhaSep ({fold_strategy})"
elif file_name == "chin":
    title = f"RNAPSEC ({fold_strategy})"
dirs = {"lgb": f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/", 

        "gp": f'../gp_opt/result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/', 
        "mlp": f'../mlp_opt/result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/',
        
        }
fig = plt.figure(figsize = (10, 10))
color = {"lgb":"darkred", "mlp": "darkblue", "gp": "forestgreen", "rf": "darkorange"}
for model_name in dirs.keys():
    file_list = sorted(glob.glob( dirs[model_name] + f"pred_cutdata_val_{min_num}_{max_num}_fold*.csv"))
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
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(auc)
        return fpr_list, tpr_list, auc_list
    print(model_name)
    #各FoldのROC曲線を
    fpr_list, tpr_list, auc_list = calc_roc(file_list)
    for fpr, tpr, auc in zip(fpr_list, tpr_list, auc_list):
        plt.plot(fpr, tpr,  color = color[model_name], alpha = 0.3, lw = 0.5,)
#全予測結果のROC曲線
result_lgb = pd.read_csv(f'./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/total_result_val_{min_num}_{max_num}.csv')
result_mlp = pd.read_csv(f'../mlp_opt/result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/total_result_val_{min_num}_{max_num}.csv')
result_gp = pd.read_csv(f'../gp_opt/result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/total_result_val_{min_num}_{max_num}.csv')
# result_rf= pd.read_csv(f'../rf_opt/result_cutdata_{file_name}/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/total_result_val_{min_num}_{max_num}.csv')
for test, model, model_name in zip([result_lgb, result_gp, result_mlp,], ["LightGBM", "GPC", "MLP", ], dirs.keys()):
    y_pred_ave=test["liquid"]
    y_test=test['actual']
    fpr_ave, tpr_ave, thresholds_ave = metrics.roc_curve(y_test,y_pred_ave)
    auc_ave = metrics.auc(fpr_ave,tpr_ave)
    plt.plot(fpr_ave, tpr_ave, label=f'{model}\nROC-AUC = %.2f'%auc_ave, alpha = 1.0,  lw = 1.8, color = color[model_name])
plt.legend( loc = "lower right", fontsize = 20)
plt.grid()
plt.xlabel("False Positive Rate", fontsize = 24)
plt.ylabel("True Positive Rate", fontsize = 24)
plt.title(f"{title}", fontsize = 32, pad = 15)
# plt.title(f"RNAPSEC (Curated)", fontsize = 32, pad = 15)
plt.show()

fig.savefig(f"./honbun_result/fig10_model_selection_{file_name}_{fold_strategy}.png")