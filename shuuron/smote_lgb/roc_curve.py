import numpy as np
import pandas as pd
import yaml
import glob

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
val_fold_strategy = args["val_fold_strategy"]
n_splits = args["n_splits"]
random_state = args["random_state"]
smote_data = args["smote_data"]
max_num = args["max_num"]
min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
val_fold = args["val_fold"]
data_dir = args["data_dir"]
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])
data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}
opt = args["opt"]
if opt == True:
    opt_name = "opt_auc"
else:
    opt_name = "opt_off"

fig = plt.figure(figsize=(10, 10))
file_list = []

file_list = sorted(glob.glob(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/pred_cutdata_val_{min_num}_{max_num}_fold*.csv"))
for i, filename in enumerate(file_list):
    test = pd.read_csv(filename, )

    y_pred=test['liquid']
    y_test=test['actual']
    #AUCを求める
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
    auc = metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr,  color = "darkgrey",alpha = 0.3, lw = 0.9)

#平均のAUCを求める
test= pd.read_csv(f'./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/total_result_val_{min_num}_{max_num}.csv')

y_pred_ave=test["liquid"]
y_test=test['actual']
fpr_ave, tpr_ave, thresholds_ave = metrics.roc_curve(y_test,y_pred_ave)
auc_ave = metrics.auc(fpr_ave,tpr_ave)

#平均のROC曲線を描く
plt.plot(fpr_ave, tpr_ave, label=f'total (area = %.2f)'%auc_ave, color = "black", alpha = 1.0,  lw = 1.2)

plt.plot(np.linspace(1, 0, len(fpr)),np.linspace(1, 0, len(fpr)), label='Ramdom ROC curve (area = %.2f)'%0.5, linestyle = '--', color = "gray", lw = 0.5)
plt.plot([0,0,1],[0,1,1], linestyle='--',label='ideal ROC curve (area = %.2f)'%1.0, color = "gray", lw = 0.5)

plt.legend(loc = "lower right", fontsize =16) 

plt.title(f"ROC {file_name}({max_num})\n smote {smote_data} {opt_name})\n{fold_strategy} val_fold = {val_fold_strategy} (er = {early_stopping_num}, val_fold = {val_fold_strategy})",  fontsize = 24, pad = 10)

plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate',  fontsize = 16)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.grid(True)
plt.show()
fig.savefig(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/roc_auc.png")

print("==========================")
print("ROC-AUC (total): ", auc_ave)
print("==========================")