import optuna
import numpy as np
import lightgbm as lgb
import pandas as pd
import yaml
import glob
import pickle

import os
import sklearn.metrics as metrics
from sklearn import preprocessing
#model
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score

import matplotlib  # <--ここを追加
matplotlib.use('Agg')  # https://python-climbing.com/runtimeerror_main_thread_is_not_in_main_loop/
from matplotlib import pyplot as plt

def main():

    #交差検証とROC曲線、valデータ付き (分割方法はconfig_.yamlで指定）、パラメーター調整なし
    with open("./config_.yaml",'r')as f:
        args = yaml.safe_load(f)

    target_label = args["target_label"]
    group_label = args["group_label"]
    ignore_features = args["ignore_features"]
    fold_strategy = args["fold_strategy"]
    val_fold_strategy = args["val_fold_strategy"]


    file_name = args["file_name"]
    mor_class = np.array(args["mor_class"])
    
    # max_num = args["max_num"]
    min_num = args["min_num"]

    val_fold = args["val_fold"]
    #model_selection

    #使用するデータの読み出し
    files =f"../data/{file_name}.csv"
    print(files)
    df = pd.read_csv(files, index_col=False) 
    df = df[df[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)

    max_num= df.shape[0]
    print("max_num", max_num)
    #X, y, groups
    if ignore_features == False:  #省くカラムがない場合   
        X = df.drop([target_label, group_label,"rnapsec_all_col_idx", "aa_rna_label", "protein_sequence", "rna_sequence", "protein_name"], axis = "columns").reset_index(drop = True).values
        assert X.shape[1] == 131, f"{df.drop([target_label, group_label,], axis = 1).columns}, {X.shape} columns false"
    else:
        print(ignore_features)
        X = df.drop([target_label, group_label], axis = "columns").reset_index(drop = True)
        X = X.drop(ignore_features, axis = "columns").reset_index(drop = True).values
    y = df[target_label].values
    groups = df[group_label].values
    print(df[group_label].unique().shape)
    mor = [["solute", "liquid", "solid"][i] for i in mor_class]

    def n_splits():
        if fold_strategy == "LeaveOneGroupOut":
            n_splits = df.shape[0]
        else:
            n_splits = args["n_splits"]
        return n_splits
    n_splits = n_splits()

    #test_train, train_valの分割方法
    def def_fold_strategy(fold_strategy, n_split = n_splits):
        if fold_strategy == "StratifiedKFold":
            kf = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = 42)        
        elif fold_strategy == "LeaveOneGroupOut":
            kf = LeaveOneGroupOut()
        return kf

    test_cv = def_fold_strategy(fold_strategy, n_split=n_splits)
    val_cv = def_fold_strategy(val_fold_strategy, n_split=val_fold)

    def objective(trial, X_train_val, y_train_val, model_name):
        # カスタムスコアラーを作成
        scorer = make_scorer(roc_auc_score, multi_class='ovr', average='macro')
        
        # モデルごとのパラメータ設定
        if model_name == "AdaBoostClassifier":
            max_depth = trial.suggest_categorical("max_depth", [5, 6, 7, 8, 9, 10, None])
            base_estimator = DecisionTreeClassifier(max_depth=max_depth)
            learning_rate = trial.suggest_categorical("learning_rate", [0.5, 1.0, 1.5, 2.0, 2.5])
            model = AdaBoostClassifier(base_estimator=base_estimator, learning_rate=learning_rate, random_state=30)
        
        elif model_name == "RandomForestClassifier":
            n_estimators = trial.suggest_categorical("n_estimators", [10, 100, 200, 300, 400, 500])
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            max_depth = trial.suggest_categorical("max_depth", [1, 2, 3, 4])
            max_features = trial.suggest_categorical("max_features", ['sqrt', 'log2', None])
            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                        max_depth=max_depth, max_features=max_features, random_state=42)
        
        elif model_name == "KNeighborsClassifier":
            n_neighbors = trial.suggest_categorical('n_neighbors', [1, 3, 5, 7, 9, 11, 15, 21])
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            p = trial.suggest_categorical('p', [1, 2])
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

        elif model_name == "GaussianNB":
            var_smoothing = trial.suggest_loguniform('var_smoothing', 1e-9, 1)
            model = GaussianNB(var_smoothing=var_smoothing)
        
        elif model_name == "LogisticRegression":
            C = trial.suggest_categorical("C", [10 ** i for i in range(-5, 6)])
            random_state = trial.suggest_categorical("random_state", [i for i in range(0, 101, 20)])
            model = LogisticRegression(C=C, random_state=random_state)
        
        elif model_name == "LGBMClassifier":
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
                # "random_state" : trial.suggest_categorical("random_state", range(1, 101, 10))
            }
            model = lgb.LGBMClassifier(boosting_type='gbdt',
                                        n_estimators=1000, **params)
        else:
            raise ValueError("Invalid model name")
        
        # ROC-AUCスコアでクロスバリデーションを計算
        scores = cross_val_score(model, X_train_val, y_train_val, cv=5, scoring=scorer)
        return np.mean(scores)

    def set_best_params(best_params):
    # 最適なパラメータを使用してモデルをトレーニング
        best_params = study.best_params

        if model_name == "AdaBoostClassifier":
            base_estimator = DecisionTreeClassifier(max_depth=best_params["max_depth"])
            model = AdaBoostClassifier(base_estimator=base_estimator, learning_rate=best_params["learning_rate"], random_state=30)

        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(
                                        random_state=30)

            model = model.set_params(**best_params)
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier()
            model = model.set_params(**best_params)
        elif model_name == "GaussianNB":
            model = GaussianNB()
            model = model.set_params(**best_params)
        elif model_name == "LogisticRegression":
            model = LogisticRegression() #C=best_params["C"],random_state=best_params["random_state"]
            model = model.set_params(**best_params)        
        elif model_name == "LGBMClassifier":
            model = lgb.LGBMClassifier(random_state = 30)
            model = model.set_params(**best_params)

        return model

  
    def standard(X = X):
        sc=preprocessing.StandardScaler()
        sc.fit(X)
        pickle.dump(sc, open(f"sc_fitted_{file_name}_{max_num}.pkl", "wb"))
        X=sc.transform(X)
        return X
    
    def makedirs(result_path):
        os.makedirs(f"{result_path}", exist_ok = True )
        return  

    for model_name in ["KNeighborsClassifier", "LogisticRegression","GaussianNB"]:
        result_path = f"./result_logocv/{model_name}"
        makedirs(result_path)
        if (model_name == "LGBMClassifier") | (model_name == "AdaBoostClassifier"):
            print("##########")
            print("without preprocessing", model_name)
            X = X
        else:
            print("##########")
            print("with preprocessing", model_name)
            X = standard()
        fig = plt.figure (figsize = (10, 10))
        aucs_list=[]
        models = {}
        probs_test_list = []
        aucs_list = []
        num_test_data ={}
        aucs_dic = {}
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []  

        auc_mean_dic = {}
        auc_all_dic = {}
        num_fold_dict = {}
        total_fold_dic = {}

        for x, (train_val_index, test_index) in enumerate(test_cv.split(X, y, groups = groups )):
            
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
            groups_train_val, group_test =  groups[train_val_index], groups[test_index]
            print("fold: ", x)
            print("X_train_val shape: ", X_train_val.shape)
            print("X_test shape: ", y_test.shape)
            #train_valをtrainとvalに分割
            train_index, val_index = next(val_cv.split(X_train_val, y_train_val,groups = groups_train_val))
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            y_train, y_val = y_train_val[train_index], y_train_val[val_index]

            # Optunaのstudyオブジェクトを作成して最適化を実行
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, X_train_val, y_train_val, model_name), n_trials=100, n_jobs=8) #n_jobs: 並列処理
            # Optunaのstudyオブジェクトを作成して最適化を実行
            best_params = study.best_trial.params #
            clf = set_best_params(best_params)
            # clf.set_params(**best_params)

            model_opt = clf.fit(X_train_val, y_train_val) #最適化したパラメーターでモデルを訓練
            
            # Save the model
            test_preds = model_opt.predict(X_test, )
            models[x] = model_opt
            probs = model_opt.predict_proba(X_test)

            probs_test = pd.DataFrame(probs, columns = mor)
            probs_test["preds"] = test_preds
            probs_test["actual"] = y_test
            probs_test["group_label"] = groups[test_index]
            probs_test.to_csv(f"{result_path}/pred_cutdata_val_{min_num}_{max_num}_fold{x}.csv", index = False)
            probs_test_list.append(probs_test)
            
            y_pred=probs_test['liquid']
            y_test=probs_test['actual']

            #testのroc
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auc = metrics.auc(fpr,tpr)
            if auc>=0:
                aucs_list.append(auc)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc)

            plt.plot(fpr, tpr, color = "darkgreen",alpha = 0.8, lw = 0.9,label = f'test group: {x} (auc = %.2f)'%auc)
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
            num_test_data[x] = X_test.shape[0]
            aucs_dic[x] = auc
        #kfold の結果を一つのリストにまとめる
        result_list=[]
        result_files = glob.glob(f"{result_path}/pred_cutdata_val_{min_num}_{max_num}_fold*.csv")
        for file in result_files:
            result_list.append(pd.read_csv(file))

        df_probs_test_list = pd.concat(probs_test_list)
        df_probs_test_list.to_csv(f'{result_path}/total_result_val_{min_num}_{max_num}.csv',index=False)
        # 各FoldのROC-AUCとtestデータ数を別ファイルで出力
        df_auc_x = pd.DataFrame()
        df_auc_x["auc"] = pd.DataFrame(aucs_dic, index=aucs_dic.keys()).iloc[0]
        df_auc_x["num_test"] = pd.DataFrame(num_test_data, index=num_test_data.keys()).iloc[0]
        df_auc_x.to_csv(f"{result_path}/roc_auc_x_fold.csv")
        # print("a")
        #cvの全モデルを保存
        with open (f"{result_path}/model_list_{file_name}.pickle", mode = "wb") as f:
            pickle.dump(models, f)
        #全テストデータのROC曲線
        auc_mean = sum(aucs_list)/len(aucs_list)
        auc_mean_dic[model_name] = auc_mean
        num_fold_dict[model_name] = len(aucs_list)
        y_pred_ave=df_probs_test_list["liquid"]
        y_test_ave=df_probs_test_list['actual']
        fpr_ave, tpr_ave, thresholds_ave = metrics.roc_curve(y_test_ave,y_pred_ave)
        auc_ave = metrics.auc(fpr_ave,tpr_ave)
        auc_all_dic[model_name] = auc_ave
        print(clf)
        print("###########")
        print(sum(aucs_list))
        print(len(aucs_list))
        print(aucs_list)
        print(sum(aucs_list)/len(aucs_list))
        print("##############")
        # plt.plot([0, 0, 0], [0,0,0], label = f"mean (area = %.2f)"%(sum(aucs_list)/len(aucs_list)),)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr,
            label=f'mean (area = %.2f)' % (mean_auc),
            lw=2, alpha=.8,)
        plt.plot(fpr_ave, tpr_ave, label=f'total (area = %.2f)'%auc_ave, color = "black",alpha = 1.0,  lw = 1.4)
        plt.plot(np.linspace(1, 0, len(fpr)),np.linspace(1, 0, len(fpr)),linestyle = '--', color = "black", lw = 1)
        plt.plot([0,0,1],[0,1,1], linestyle='--', color = "black", lw = 1.2)
        plt.legend(fontsize = 18, loc = "lower right") 
        plt.title(f"{model_name}_{file_name}: ROC {min_num}_{max_num} {fold_strategy}\n val_fold = {val_fold_strategy})",  fontsize = 12, pad = 10)
        plt.xlabel('False Positive Rate', fontsize = 24)
        plt.ylabel('True Positive Rate',  fontsize = 24)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.grid(True)
        # plt.show()   

        # total_fold_dic[model]= len(np.unique(groups))
        total_fold_dic[model_name]= x+1
        fig.savefig(f"{result_path}/roc_val_train_test.png")
        plt.close()
        print("=============================")
        print("data: ", files)
        print("min_num, max_num:", min_num, max_num)
        print("train_val/test: ", fold_strategy)
        print("train/val: ", val_fold_strategy)
        print("============================")
        print("TEST fold strategy: ", test_cv)
        print("VAL fold strategy: ", val_cv)
        print("X_shape", X.shape)
        print("n_splits: ", n_splits)

        df_auc = pd.DataFrame()
        df_auc["all_roc"] = pd.DataFrame(auc_all_dic, index=auc_all_dic.keys()).iloc[0]
        df_auc["mean_roc"] = pd.DataFrame(auc_mean_dic, index=auc_mean_dic.keys()).iloc[0]
        df_auc["fold_notna"] = pd.DataFrame(num_fold_dict, index=num_fold_dict.keys()).iloc[0]
        df_auc["total_fold_num"] =  pd.DataFrame(total_fold_dic, index=total_fold_dic.keys()).iloc[0]
        df_auc.to_csv(f"{result_path}/logocv_score.csv")


        with open(f"{result_path}/config.yaml", "wb") as tf:
            pickle.dump(args,tf)

if __name__ == "__main__":
    main()
