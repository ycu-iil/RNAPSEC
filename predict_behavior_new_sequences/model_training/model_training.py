import optuna
import numpy as np
import lightgbm as lgb
import pandas as pd
import yaml
import pickle
#model
from sklearn.model_selection import StratifiedKFold, cross_val_score

import matplotlib  # <--ここを追加
matplotlib.use('Agg')  # https://python-climbing.com/runtimeerror_main_thread_is_not_in_main_loop/
from matplotlib import pyplot as plt

#交差検証とROC曲線、valデータ付き (分割方法はconfig_.yamlで指定）、パラメーター調整なし
with open("./config_model.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])

files =f"../../data/{file_name}.csv"
print(files)
df = pd.read_csv(files, index_col=False) 


df = df[df[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)

max_num= df.shape[0]
print("max_num", max_num)
#X, y, groups
if ignore_features == False:  #省くカラムがない場合   
    X = df.drop([target_label, group_label,"rnapsec_all_col_idx", "aa_rna_label", "protein_sequence", "rna_sequence", "protein_name"], axis = "columns").reset_index(drop = True).values
    assert X.shape[1] == 131, f"{df.drop([target_label, group_label], axis = 1).columns}, {X.shape} columns false"
else:
    print(ignore_features)
    X = df.drop([target_label, group_label], axis = "columns").reset_index(drop = True)
    X = X.drop(ignore_features, axis = "columns").reset_index(drop = True).values
y = df[target_label].values
groups = df[group_label].values
print(df[group_label].unique().shape)


def objective(trial):
    params = { #epoch 
        'objective': 'binary',
        'metric': 'binary_logloss',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        "random_state" : trial.suggest_categorical("random_state", range(1, 101, 10))
    }
    clf = lgb.LGBMClassifier(boosting_type='gbdt',
                                        n_estimators=1000, **params)
    clf.set_params(**params)
    scores = cross_val_score(clf, X=X, y=y,cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True),scoring='roc_auc')
    return scores.mean()
seed = 0
study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=100)
best_params = study.best_trial.params
print(best_params)
clf = lgb.LGBMClassifier()
clf.set_params(**best_params)
model_opt = clf.fit(X, y)
with open (f"../pretrained_model.pickle", mode = "wb") as f:
    pickle.dump(model_opt, f)
