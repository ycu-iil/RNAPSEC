
import numpy as np
import pandas as pd
import yaml
import pickle
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

#引数指定
with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)
target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
mor_class = np.array(args["mor_class"])
file_name = args["file_name"]
min_num = args["min_num"]

# 最大値の指定
def def_max_num():
    df = pd.read_csv( f"../data/{file_name}.csv", index_col=False)
    df = df[df[target_label].isin(mor_class)].drop_duplicates().reset_index(drop = True) #予測する形態指定(solute, liquid, solid)
    return df.shape[0]
max_num = def_max_num()

d = 230817
def calc(model_name, kf_state):
    result_path = f"./result_repeated_{fold_strategy}/{model_name}/kf_state_{kf_state}/"
    # 分割したテストデータの読み出し
    result = []
    file_path = f"./result_repeated_StratifiedGroupKFold/split_kf_state_{kf_state}"
    for i in range(10):
        df = pd.read_csv(f"{file_path}/test_df_{group_label}_fold{i}.csv",index_col = False)
        result.append(df)
    df_rnapsec = pd.concat(result, axis = "index")
    print(df_rnapsec.columns)
    df_rnapsec["ex_label"] = df_rnapsec.aa_rna_label
    #モデルの読み出し
    with open(f"{result_path}/model_list_{file_name}.pickle", 'rb') as web:
        model_rnapsec = pickle.load(web)
    def create_test_ex(df):
        protein_conc = np.linspace(df.protein_conc_log.min()-1, df.protein_conc_log.max()+1, 20) #0629:間隔 20 ->0.2 
        protein_conc.shape
        rna_conc = np.linspace(df.rna_conc_log.min()-1, df.rna_conc_log.max()+1, 20) #0629:間隔 20 ->0.2 
        rna_conc.shape
        #データセット中の該当するデータのタンパク質濃度とRNA濃度の分布の範囲内をカバーする濃度値を設定
        patterns = []
        for protein in protein_conc:
            for rna in rna_conc:
                patterns.append((protein,rna))
        df_add= pd.DataFrame(patterns)
        df_add = df_add.rename(columns = {0:"protein_conc_log", 1:"rna_conc_log"})
        df_test = pd.concat([df_add, df], axis = "index").reset_index(drop = True)    
        return df_test

    #出力する図の保存先
    os.makedirs(f"{result_path}/souzu_{d}/", exist_ok = True)
    num_list = {}
    test_list = {}
    import matplotlib as mpl
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f%%'))#y軸小数点以下3桁表示
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f%%'))#y軸小数点以下3桁表示
    
    for i in range(10):
        df_fold = df_rnapsec[df_rnapsec.fold_num == i]
        for group in df_fold[group_label].unique():
            df_target_group = df_fold[df_fold[group_label] == group]
            num_list[group] = []
            test_list[group] = {}
            result_list = [] #group ごとにcsv出力
            for aa_rna in df_fold[df_fold[group_label] == group].ex_label.unique():
                print("aa_label:", group)
                print("ex_label:", aa_rna)
                df = pd.DataFrame() #保存用
                df_test = pd.DataFrame() #拡張用
                num_list[group].append(aa_rna)
                df_target = df_target_group[df_target_group.ex_label==aa_rna]
                
                df_test = create_test_ex(df_target) #テストデータ拡張
                target_x = df_target.drop([target_label, group_label, "rnapsec_all_col_idx",
        'protein_sequence', 'rna_sequence', 'protein_name', "fold_num",
        'aa_rna_label', "fold_num", "ex_label"], axis = "columns")
                df_test.iloc[:, 2:] = df_test.iloc[:, 2:].fillna(target_x.drop(['protein_conc_log', "rna_conc_log"], axis = "columns").iloc[0, :]) #溶質濃度以外の条件は複製する
                df_test = df_test.loc[:, df_target.columns]
                df_test = df_test[df_test.mor_label.isna()]
                test_list[group][aa_rna] = df_test.copy()

                #増やしたデータを予測
                X = df_test.drop([target_label, group_label, "rnapsec_all_col_idx",
        'protein_sequence', 'rna_sequence', 'protein_name', "fold_num",
        'aa_rna_label', "fold_num", "ex_label"], axis = "columns").values
                print("X.shape: ", X.shape)
                print(X.shape)
                assert X.shape[1] == 131, "input columns != 131" #特徴量の数が一致しているか

                df_test["preds"] = model_rnapsec[i].predict(X)
                df_test["proba"] = model_rnapsec[i].predict_proba(X)[:, 1]
                
                result_list.append(df_test)
                test_list[group][aa_rna] = df_test.copy()
                pred_protein_1 = df_test[(df_test.preds == 1)].protein_conc_log
                pred_rna_1 = df_test[(df_test.preds == 1)].rna_conc_log
                pred_protein_0 = df_test[(df_test.preds == 0)].protein_conc_log
                pred_rna_0 = df_test[(df_test.preds == 0) ].rna_conc_log
                act_protein_1 = df_target[(df_target.mor_label == 1)].protein_conc_log
                act_rna_1 = df_target[(df_target.mor_label == 1)].rna_conc_log
                act_protein_0 = df_target[(df_target.mor_label == 0)].protein_conc_log
                act_rna_0 = df_target[(df_target.mor_label == 0)].rna_conc_log
                
                fig, ax = plt.subplots(figsize = (10, 10))
                # plot_setting()
                ax.scatter(x = pred_protein_1, y = pred_rna_1, c = "lightskyblue",alpha = 1, label = "Prediction: Liquid", s = 620, marker = "s") # c = "dodgerblue"
                ax.scatter(x = pred_protein_0, y = pred_rna_0, c = "navajowhite", alpha = 1, label = "Prediction: Solute", s =620, marker = "s")
                ax.scatter(x = act_protein_1, y = act_rna_1, c = "mediumblue", alpha = 1, marker = "D", label = "Experiment: Liquid", s = 200)
                ax.scatter(x = act_protein_0, y = act_rna_0,  c = "firebrick",marker = "D", alpha = 1,  label = "Experiment: Solute", s =200)
                ax.set_xlabel("Protein conc. (log μM)", fontsize = 26, labelpad = 15)
                ax.set_ylabel("RNA conc. (log μM)", fontsize = 26, labelpad = 15)
                ax.xaxis.set_major_locator(ticker.LinearLocator(6))
                ax.xaxis.set_tick_params(direction='out', labelsize=20, width=3, pad=10)
                ax.yaxis.set_tick_params(direction='out', labelsize = 20, width=3, pad=10)
                ax.legend(fontsize = 28,  loc='lower right', bbox_to_anchor=(1.8, 0))
                ax.set_title(f"{model_name} Random_state = {kf_state}\nGroup = {group}, EX = {aa_rna}", fontsize = 26, pad = 24)
                fig.savefig(f"{result_path}/souzu_{d}/souzu_group{group}_{aa_rna}.png", bbox_inches='tight', transparent=False)
                plt.close() 
            os.makedirs(f"{result_path}/souzu_result_{d}/", exist_ok=True)
            if len(result_list) >0:
                print(group)
                print(len(result_list))
                pd.concat(result_list, axis="index").to_csv(f"{result_path}/souzu_result_{d}/souzu_result_{group}.csv", index=False) #僧都書くときに使ったデータを保存
    return

model_name = "AdaBoostClassifier"
kf_state = 90
calc(kf_state=kf_state, model_name=model_name)