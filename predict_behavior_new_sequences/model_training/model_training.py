import optuna
import numpy as np
import lightgbm as lgb
import pandas as pd
import yaml
import pickle
#model
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
    max_depth = trial.suggest_categorical("max_depth", [5, 6, 7, 8, 9, 10, None])
    base_estimator = DecisionTreeClassifier(max_depth=max_depth)
    learning_rate = trial.suggest_categorical("learning_rate", [0.5, 1.0, 1.5, 2.0, 2.5])
    random_state = trial.suggest_categorical("random_state", range(1, 101, 10))
    clf = AdaBoostClassifier(base_estimator=base_estimator, learning_rate=learning_rate, random_state=random_state)
    scores = cross_val_score(clf, X=X, y=y,cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True),scoring='roc_auc')
    return scores.mean()
seed = 0
study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=100)
best_params = study.best_trial.params

base_estimator = DecisionTreeClassifier(max_depth=best_params["max_depth"])
clf = AdaBoostClassifier(base_estimator=base_estimator, learning_rate=best_params["learning_rate"], random_state=best_params["random_state"])
model_opt = clf.fit(X, y)
with open (f"../pretrained_model.pickle", mode = "wb") as f:
    pickle.dump(model_opt, f)
