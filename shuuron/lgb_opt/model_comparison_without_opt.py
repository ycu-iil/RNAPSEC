import argparse
import numpy as np
import pandas as pd
import yaml
import glob
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut, cross_val_score, cross_validate

import pickle

import os
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

#model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, Lasso, LogisticRegression

from sklearn import preprocessing


#交差検証とROC曲線、valデータ付き (分割方法はconfig_.yamlで指定）、パラメーター調整なし
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
#model_selection
# model = args["model"] #forで回す

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

# データの整形
sc=preprocessing.StandardScaler()
sc.fit(X)
X=sc.transform(X)

#test_train, train_valの分割方法
def def_fold_strategy(fold_strategy, n_split = n_splits):
    if fold_strategy == "GroupKFold":
        kf = GroupKFold(n_splits = n_split)
    # elif fold_strategy == "StratifiedGroupKFold":
    #     kf = StratifiedGroupKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    elif fold_strategy == "KFold":
        kf = KFold(n_splits = n_split, shuffle = True, random_state = random_state)
    elif fold_strategy == "StratifiedKFold":
        kf = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = random_state)        
    elif fold_strategy == "LeaveOneGroupOut":
        kf = LeaveOneGroupOut()
    return kf

test_cv = def_fold_strategy(fold_strategy, n_split=n_splits)
val_cv = def_fold_strategy(val_fold_strategy, n_split=val_fold)


probs_test_list = []
aucs_list = []

models = {}
cm_list = []
# for model in ["RandomForestClassifier", "DecisionTreeClassifier", "AdaBoostClassifier", "Lasso", "LogisticRegression", "KNeighborsClassifier", "GaussianNB", "QuadraticDiscriminantAnalysis", "SVC"]:
for model in ["RandomForestClassifier", "DecisionTreeClassifier", "AdaBoostClassifier", "LogisticRegression", "KNeighborsClassifier", "GaussianNB", "QuadraticDiscriminantAnalysis"]:
    def makedirs():
        os.makedirs(f"./result_cutdata_{file_name}/", exist_ok = True)
        os.makedirs(f"./result_cutdata_{file_name}/{model}/", exist_ok = True)
        os.makedirs(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/", exist_ok = True)
        os.makedirs(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/", exist_ok = True)
        os.makedirs(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/opt_false/", exist_ok = True)
        os.makedirs(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/opt_false/cv_{fold_strategy}_valsplit_{val_fold_strategy}/", exist_ok = True )
        os.makedirs(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/opt_false/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/", exist_ok = True )
        return  
    if model == "RandomForestClassifier":
        clf = RandomForestClassifier(random_state = 42)
        makedirs()
    elif model == "DecisionTreeClassifier" :
        clf = DecisionTreeClassifier(random_state = 42)
        makedirs()
    elif model == "KNeighborsClassifier":
        clf = KNeighborsClassifier() 
        makedirs()
    elif model == "SVC":
        clf = SVC()
        print(model)
        makedirs()
    elif model == "AdaBoostClassifier":
        print(model)
        clf = AdaBoostClassifier()
        makedirs()
    elif model == "GaussianNB":
        print(model)
        clf = GaussianNB()
        makedirs()
    elif model == "QuadraticDiscriminantAnalysis":
        print(model)
        clf = QuadraticDiscriminantAnalysis()
        makedirs()
    elif model == "SGDClassifier":
        print(model)
        clf = SGDClassifier(loss = "hinge")
        makedirs()
    elif model == "Lasso":
        print(model)
        clf = Lasso()
        makedirs()
    elif model == "LogisticRegression":
        print(model)
        clf = LogisticRegression()
        makedirs()
    else:
        print(model)
        clf = None
        continue
        
    fig = plt.figure (figsize = (10, 10))
    
    for x, (train_val_index, test_index) in enumerate(test_cv.split(X, y, groups = groups )):
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]
        groups_train_val, group_test =  groups[train_val_index], groups[test_index]
        
        #train_valをtrainとvalに分割
        train_index, val_index = next(val_cv.split(X_train_val, y_train_val,groups = groups_train_val))
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]


        clf.fit(X_train, y_train)
        train_preds = clf.predict(X_train)
        val_preds = clf.predict(X_val, )
        test_preds = clf.predict(X_test, )
        models[x] = clf

        probs = clf.predict_proba(X_test)

        probs_test = pd.DataFrame(probs, columns = mor)
        probs_test["preds"] = test_preds
        probs_test["actual"] = y_test
        probs_test["group_label"] = groups[test_index]
        
        probs_test.to_csv(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/opt_false/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/pred_cutdata_val_{min_num}_{max_num}_fold{x}.csv", index = False)
        probs_test_list.append(probs_test)
        
        y_pred=probs_test['liquid']
        y_test=probs_test['actual']

        #testのroc
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fpr,tpr)
        aucs_list.append(auc)
        
        plt.plot(fpr, tpr, color = "grey",alpha = 0.3, lw = 0.9,)
        print("test_auc:", metrics.auc(fpr, tpr))
        probs_train = clf.predict_proba(X_train)
        #trainのROC
        fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train, probs_train[:,1])
        plt.plot(fpr_train, tpr_train, color = "red",lw =0.9)
        print("train_auc:", metrics.auc(fpr_train,tpr_train))
        #valのROC
        probs_val = clf.predict_proba(X_val)
        fpr_val, tpr_val, thresholds = metrics.roc_curve(y_val, probs_val[:,1])
        plt.plot(fpr_val, tpr_val, color = "blue",lw =0.9)
        print("val_auc:", metrics.auc(fpr_val,tpr_val))
        
    #kfold の結果を一つのリストにまとめる
    result_list=[]
    result_files = glob.glob(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/opt_false/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/pred_cutdata_val_{min_num}_{max_num}_fold*.csv")
    for file in result_files:
        result_list.append(pd.read_csv(file))

    df = pd.concat(probs_test_list)
    df.to_csv(f'./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/opt_false/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/total_result_val_{min_num}_{max_num}.csv',index=False)
    # print("a")
    #cvの全モデルを保存
    with open (f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/opt_false/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/model_list_{file_name}.pickle", mode = "wb") as f:
        pickle.dump(models, f)
    #全テストデータのROC曲線
    y_pred_ave=df["liquid"]
    y_test_ave=df['actual']
    fpr_ave, tpr_ave, thresholds_ave = metrics.roc_curve(y_test_ave,y_pred_ave)
    auc_ave = metrics.auc(fpr_ave,tpr_ave)
    print(clf)
    plt.plot(fpr_ave, tpr_ave, label=f'total (area = %.2f)'%auc_ave, color = "black",alpha = 1.0,  lw = 1.4)
    plt.plot(np.linspace(1, 0, len(fpr)),np.linspace(1, 0, len(fpr)),linestyle = '--', color = "black", lw = 1)
    plt.plot([0,0,1],[0,1,1], linestyle='--', color = "black", lw = 1.2)
    plt.legend(fontsize = 18, loc = "lower right") 
    plt.title(f"{model}: ROC {min_num}_{max_num} {fold_strategy}\n val_fold = {val_fold_strategy} (er = {early_stopping_num}, val_fold = {val_fold_strategy})",  fontsize = 12, pad = 10)
    plt.xlabel('False Positive Rate', fontsize = 24)
    plt.ylabel('True Positive Rate',  fontsize = 24)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.grid(True)
    plt.show()

    fig.savefig(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/opt_false/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/roc_val_train_test.png")
    print("=============================")
    print("data: ", files)
    print("min_num, max_num:", min_num, max_num)
    print("train_val/test: ", fold_strategy)
    print("train/val: ", val_fold_strategy)
    print("============================")


