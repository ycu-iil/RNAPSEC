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
import optuna
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

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
fig = plt.figure (figsize = (10, 10))
models = {}
cm_list = []

for x, (train_val_index, test_index) in enumerate(test_cv.split(X, y, groups = groups )):
    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]
    groups_train_val, group_test =  groups[train_val_index], groups[test_index]
    
    #train_valをtrainとvalに分割
    train_index, val_index = next(val_cv.split(X_train_val, y_train_val,groups = groups_train_val))
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]


    # params = {'objective': 'binary', 
    #     'metric': 'logloss',  
    #     'random_state': 42,  
    #     'boosting_type': 'gbdt', 
    #     'verbose': -1
    #     }
    # clf = lgb.LGBMClassifier( **params)   
    # clf =clf.fit(X_train, y_train,
    #             eval_metric='logloss',
    #             eval_set=[(X_val, y_val)],
    #             early_stopping_rounds= early_stopping_num, 
    #             verbose=0)

    
    # def objective(trial):
    #     params = {
    #         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
    #         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    #         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    #         'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
    #         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
    #         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    #         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    #         # "n_estimators": trial.suggest_int("n_estimators", 50, 150)
    #     }
    #     clf.set_params(**params)
    #     # scores = cross_val_score(clf, X_train_val, y_train_val, cv = GroupKFold(n_splits= 5),     
    #     #                         scoring='roc_auc', fit_params=fit_params, n_jobs=-1, groups = group_train_val)
    #     clf =clf.fit(X_train, y_train,
    #             eval_metric='logloss',
    #             eval_set=[(X_val, y_val)],
    #             early_stopping_rounds= early_stopping_num, 
    #             verbose=0)
    #     proba = clf.predict_proba(X_test)
    #     score = metrics.roc_auc_score(proba, y_test)
    #     return score
    def objective(trial):
        params = {
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            # "n_estimators": trial.suggest_int("n_estimators", 50, 150)
        }
        clf.set_params(**params)
        # scores = cross_val_score(clf, X_train_val, y_train_val, cv=GroupKFold(n_splits=5),
        #                         scoring='roc_auc', fit_params=fit_params, n_jobs=-1, groups = groups_train_val)
        scores = cross_val_score(clf, X_train_val, y_train_val, cv=val_cv,
                                scoring='roc_auc', fit_params=fit_params, n_jobs=-1, groups = groups_train_val)
        return scores.mean()
    seed = 0
    clf = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',
                    random_state=seed, n_estimators=100)
    fit_params = {
        'verbose': 0,
        'early_stopping_rounds': 10,
        'eval_metric': 'roc_auc',
        'eval_set': [(X_val, y_val)]
        }

    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=50)
    best_params = study.best_trial.params
    clf.set_params(**best_params)
    clf.fit(X_train, y_train)
    #train, val, testの予測値の評価
    # print("###########################################")
    # print(f"{x} fold train clf reports:", classification_report(y_train, train_preds))
    # print("val clf reports:", classification_report(y_val, val_preds))
    # print("test auc:", classification_report(y_test, test_preds))
    # print("##################")
    # print("test confusion matrix")
    # cm = confusion_matrix(y_test, test_preds)
    # cm_list.append(cm)
    # print(cm)
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val, )
    test_preds = clf.predict(X_test, )
    models[x] = clf

    probs = clf.predict_proba(X_test)

    probs_test = pd.DataFrame(probs, columns = mor)
    probs_test["preds"] = test_preds
    probs_test["actual"] = y_test
    probs_test["group_label"] = groups[test_index]
    
    probs_test.to_csv(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/pred_cutdata_val_{min_num}_{max_num}_fold{x}.csv", index = False)
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
result_files = glob.glob(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/pred_cutdata_val_{min_num}_{max_num}_fold*.csv")
for file in result_files:
    result_list.append(pd.read_csv(file))

df = pd.concat(probs_test_list)
df.to_csv(f'./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/total_result_val_{min_num}_{max_num}.csv',index=False)
# print("a")
#cvの全モデルを保存
with open (f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/model_list_{file_name}.pickle", mode = "wb") as f:
    pickle.dump(models, f)
#全テストデータのROC曲線
y_pred_ave=df["liquid"]
y_test_ave=df['actual']
fpr_ave, tpr_ave, thresholds_ave = metrics.roc_curve(y_test_ave,y_pred_ave)
auc_ave = metrics.auc(fpr_ave,tpr_ave)

plt.plot(fpr_ave, tpr_ave, label=f'total (area = %.2f)'%auc_ave, color = "black",alpha = 1.0,  lw = 1.4)
plt.plot(np.linspace(1, 0, len(fpr)),np.linspace(1, 0, len(fpr)),linestyle = '--', color = "black", lw = 1)
plt.plot([0,0,1],[0,1,1], linestyle='--', color = "black", lw = 1.2)
plt.legend(fontsize = 18, loc = "lower right") 
plt.title(f"ROC {min_num}_{max_num} {fold_strategy}\n val_fold = {val_fold_strategy} (er = {early_stopping_num}, val_fold = {val_fold_strategy})",  fontsize = 12, pad = 10)
plt.xlabel('False Positive Rate', fontsize = 24)
plt.ylabel('True Positive Rate',  fontsize = 24)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.grid(True)
plt.show()

fig.savefig(f"./result_cutdata_{file_name}/data_{min_num}_{max_num}/opt/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/opt_auc/roc_val_train_test.png")
print("=============================")
print("data: ", files)
print("min_num, max_num:", min_num, max_num)
print("train_val/test: ", fold_strategy)
print("train/val: ", val_fold_strategy)
print("============================")


