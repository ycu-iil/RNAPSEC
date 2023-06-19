#学習データ(val, train)にsmoteを適応
import argparse
import numpy as np
from imblearn.over_sampling import SMOTE
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
#trainだけsmoteかける。validationとtestはかけない
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
df_chin = df_chin.dropna()

#X, y, groups
if ignore_features == False:  #省くカラムがない場合   
    X = df_chin.drop([target_label,group_label], axis = "columns").reset_index(drop = True).values #group込みのX, 
else:
    X = df_chin.drop([target_label, group_label, ignore_features], axis = "columns").reset_index(drop = True).values
y = df_chin[target_label].values
groups = df_chin[group_label].values
os.makedirs(f"./result_cutdata_{file_name}/", exist_ok = True)
os.makedirs(f"./result_cutdata_{file_name}/smote_trainvaldata/", exist_ok = True)
os.makedirs(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/", exist_ok = True)

os.makedirs(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/", exist_ok = True)
os.makedirs(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/", exist_ok = True )
os.makedirs(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/", exist_ok = True )

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
# val_cv = def_fold_strategy(val_fold_strategy, n_split=val_fold)
val_cv = StratifiedKFold(shuffle=True, random_state=42, n_splits=5)


probs_test_list = []
aucs_list = []
fig = plt.figure (figsize = (10, 10))
models = {}
cm_list = []
os.makedirs(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/data_smoted", exist_ok=True)
for x, (train_val_index, test_index) in enumerate(test_cv.split(X, y, groups = groups )):
    sm = SMOTE(random_state=2)
    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]
    group_train_val = df_chin.group_label.values[train_val_index]
    #train_valをtrainとvalに分割
    X_train_val_res, y_train_val_res = sm.fit_sample(X_train_val, y_train_val.ravel())
    df_res = pd.concat([pd.DataFrame(X_train_val_res), pd.DataFrame(y_train_val_res)], axis="columns")
    df_res.to_csv(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/data_smoted/train_smoted.csv")

    train_index, val_index = next(val_cv.split(X_train_val_res, y_train_val_res))
    X_train, X_val = X_train_val_res[train_index], X_train_val_res[val_index,]
    y_train, y_val = y_train_val_res[train_index], y_train_val_res[val_index]

    print(X_train.shape, X_test.shape, X_val.shape)
    print("training: ", pd.DataFrame(y_train).value_counts())
##################optuna
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
        scores = cross_val_score(clf, X_train_val_res, y_train_val_res, cv = StratifiedKFold(shuffle=True, n_splits=5, random_state=42),
                                scoring='roc_auc', fit_params=fit_params, n_jobs=-1)
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

    # Save best parameters
    with open(f'./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/data_smoted/best_parameters.pkl', mode='wb') as f:
        pickle.dump(best_params, f)

    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test, )

    models[x] = clf
    probs = clf.predict_proba(X_test)

    probs_test = pd.DataFrame(probs, columns = mor)
    probs_test["preds"] = test_preds
    probs_test["actual"] = y_test
    probs_test["group_label"] = groups[test_index]
    
    probs_test.to_csv(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/pred_cutdata_val_{min_num}_{max_num}_fold{x}.csv", index = False)
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
result_files = glob.glob(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/pred_cutdata_val_{min_num}_{max_num}_fold*.csv")
for file in result_files:
    result_list.append(pd.read_csv(file))

df = pd.concat(probs_test_list)
df.to_csv(f'./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/total_result_val_{min_num}_{max_num}.csv',index=False)
# print("a")
#cvの全モデルを保存
with open (f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/model_list_{file_name}.pickle", mode = "wb") as f:
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
plt.title(f"ROC SMOTE (train_val, opt = on) {min_num}_{max_num} {fold_strategy}\n val_fold = {val_fold_strategy} (er = {early_stopping_num}, val_fold = {val_fold_strategy})",  fontsize = 12, pad = 10)
plt.xlabel('False Positive Rate', fontsize = 24)
plt.ylabel('True Positive Rate',  fontsize = 24)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.grid(True)
plt.show()


fig.savefig(f"./result_cutdata_{file_name}/smote_trainvaldata/opt_auc/data_{min_num}_{max_num}/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_val_fold_{val_fold}/roc_val_train_test.png")
print("=============================")
print("data: ", files)
print("min_num, max_num:", min_num, max_num)
print("train_val/test: ", fold_strategy)
print("train/val: ", val_fold_strategy)
print("============================")