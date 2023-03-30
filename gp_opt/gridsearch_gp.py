import argparse
import numpy as np
import pandas as pd
import yaml
import glob
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut, cross_val_score, cross_validate
import lightgbm as lgb
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
val_fold_strategy = args["val_fold_strategy"]
n_splits = args["n_splits"]
random_state = args["random_state"]
data_dir = args["data_dir"]
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])
data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}

max_num = args["max_num"]
min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
val_fold = args["val_fold"]
files_path = data_dir + file_name 


# for idx, name in enumerate(df_name):
files =f"{files_path}/sel_{file_name}_{min_num}_{max_num}.csv"
df_chin = pd.read_csv(files, index_col=False) 

mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_chin = df_chin[df_chin[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
#X, y, groups
if ignore_features == False:  #省くカラムがない場合   
    X = df_chin.drop([target_label, group_label], axis = "columns").reset_index(drop = True).values
else:
    X = df_chin.drop([target_label, group_label, ignore_features], axis = "columns").reset_index(drop = True).values
y = df_chin[target_label].values
groups = df_chin[group_label].values

os.makedirs(f"./result_cutdata_{file_name}/", exist_ok = True)
os.makedirs(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/", exist_ok = True)
os.makedirs(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/", exist_ok = True)
os.makedirs(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/", exist_ok = True )
os.makedirs(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/", exist_ok = True )
os.makedirs(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc", exist_ok = True )
def def_fold_strategy(fold_strategy):
    if fold_strategy == "GroupKFold":
        kf = GroupKFold(n_splits = n_splits)
    # elif fold_strategy == "StratifiedGroupKFold":
    #     kf = StratifiedGroupKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    elif fold_strategy == "KFold":
        kf = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    elif fold_strategy == "StratifiedKFold":
        kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)        
    elif fold_strategy == "LeaveOneGroupOut":
        kf = LeaveOneGroupOut()
    return kf

print(file_name)
print(fold_strategy)
print(val_fold_strategy)
print(val_fold)
print(file_name)
print(max_num)


test_cv = def_fold_strategy(fold_strategy)    
probs_test_list = []
aucs_list = []
fig = plt.figure (figsize = (10, 10))
models = []

cm_list = []

#CV
for x, (train_index, test_index) in enumerate(test_cv.split(X, y, groups = groups )):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    group_train, group_test =  groups[train_index], groups[test_index]

    clf = GaussianProcessClassifier()   
    
    params4 = {"max_iter_predict": [100, 500, 1000, 1500,  2000],
    'n_restarts_optimizer': [0, 5, 10, 15],}
    gscv = GridSearchCV(clf, params4, scoring='roc_auc', cv = 5)
    result = gscv.fit(X_train, y_train)
    print(result.best_score_)
    print(result.best_params_)

    clf = clf.set_params(**result.best_params_)
    clf =clf.fit(X_train, y_train,)

    train_preds = clf.predict(X_train)

    test_preds = clf.predict(X_test, )
    #train, val, testの予測値の評価
    print("###########################################")
    print(f"{x} fold train clf reports:", classification_report(y_train, train_preds))
    print("test auc:", classification_report(y_test, test_preds))
    print("##################")
    print("test confusion matrix")
    cm = confusion_matrix(y_test, test_preds)
    cm_list.append(cm)
    print(cm)
    
    models.append(clf)

    probs = clf.predict_proba(X_test)

    probs_test = pd.DataFrame(probs, columns = mor)
    probs_test["preds"] = test_preds
    probs_test["actual"] = y_test
    probs_test["group_label"] = groups[test_index]
    probs_test.to_csv(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/pred_cutdata_val_{min_num}_{max_num}_fold{x}.csv", index = False)
    probs_test_list.append(probs_test)
    
#kfold の結果を一つのリストにまとめる
result_list=[]
result_files = glob.glob(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/pred_cutdata_val_{min_num}_{max_num}_fold*.csv")
for file in result_files:
    result_list.append(pd.read_csv(file))

df = pd.concat(probs_test_list)
df.to_csv(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/total_result_val_{min_num}_{max_num}.csv",index=False)
print("a")