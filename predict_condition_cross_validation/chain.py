#%%
import argparse
import numpy as np
import pandas as pd
import yaml
import glob
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut, cross_val_score, cross_validate
import lightgbm as lgb
import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import ClassifierChain
import scipy.sparse as sp
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from matplotlib import cm

with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
data_dir = args["data_dir"]
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])

min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
files_path = data_dir + file_name 


files =f"../data/{file_name}.csv"
print(files)

df_rnapsec = pd.read_csv(files, index_col=False) 

mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_rnapsec = df_rnapsec[df_rnapsec[target_label].isin(mor_class)].reset_index(drop=True) #予測する形態指定(solute, liquid, solid)
max_num= df_rnapsec.shape[0]

#溶質濃度
def log_cut_bins(df):

    def cut_bins_class(y_series_1):
        rna_sc_log_class=pd.DataFrame(y_series_1)
        bins = []
        percentiles = [0, 20, 40, 60, 80]
        for i in percentiles:
            bins.append(np.percentile(y_series_1.unique(), i))
        bins.append(np.inf)

        bins_names=[0, 1, 2, 3, 4]
        rna_sc_log_class=pd.cut(rna_sc_log_class.iloc[:,0], bins ,labels=bins_names,right=False)
        rna_sc_log_class=rna_sc_log_class.to_frame()
        rna_sc_log_class=rna_sc_log_class.rename(columns={f"{y_series_1}": f"{y_series_1}_class"})
        return  rna_sc_log_class, bins
    #log_concを5クラスに分類
    rna_conc_log_class, bins_rna = cut_bins_class(df.rna_conc_log)
    rna_conc_log_class = rna_conc_log_class.rename(columns={"rna_conc_log": "rna_conc_log_class"})
    protein_conc_log_class, bins_protein = cut_bins_class(df.protein_conc_log)
    protein_conc_log_class = protein_conc_log_class.rename(columns={"protein_conc_log": "protein_conc_log_class"})
    df["rna_conc_log_class"] = rna_conc_log_class
    df["protein_conc_log_class"] = protein_conc_log_class
    df["protein_conc_log_class"][df.protein_conc_log == (np.inf)] = 4
    df.loc.__setitem__(((df.rna_conc_log == (-np.inf)), "rna_conc_log_class"), 0)
    df.loc.__setitem__(((df.rna_conc_log == (np.inf)), "rna_conc_log_class"), 4)
    return df
#pHをビン切り
def cut_ph_class(y_series_1):
    rna_sc_log_class=pd.DataFrame(y_series_1)
    bins = [0, 7, 8, 1000]
    bins_names=[0, 1, 2]
    rna_sc_log_class=pd.cut(rna_sc_log_class.iloc[:,0],bins ,labels=bins_names,right=False)
    rna_sc_log_class=rna_sc_log_class.to_frame()
    rna_sc_log_class=rna_sc_log_class.rename(columns={f"{y_series_1}": f"{y_series_1}_class"})
    return  rna_sc_log_class, bins
#tempをビン切り
def cut_bins_class(y_series_1, bins_list = [0, 25, 38, 1000]):
    rna_sc_log_class=pd.DataFrame(y_series_1)
    bins = bins_list

    bins_names=[0, 1, 2]
    rna_sc_log_class=pd.cut(rna_sc_log_class.iloc[:,0],bins ,labels=bins_names,right=False)
    rna_sc_log_class=rna_sc_log_class.to_frame()
    rna_sc_log_class=rna_sc_log_class.rename(columns={f"{y_series_1}": f"{y_series_1}_bins"})
    return  rna_sc_log_class, bins
def cut_is_bins(df, target_series):

    def cut_bins_class(y_series_1):
        df_is=pd.DataFrame(y_series_1)
        bins = []
        percentiles = [0, 20, 40, 60, 80]
        for i in percentiles:
            bins.append(np.percentile(y_series_1.unique(), i))
        bins.append(np.inf)

        bins_names=[0, 1, 2, 3, 4]
        df_is=pd.cut(df_is.iloc[:,0], bins ,labels=bins_names,right=False)
        df_is=df_is.to_frame()
        df_is=df_is.rename(columns={f"{y_series_1}": f"{y_series_1}_class"})
        return  df_is, bins
    #log_concを5クラスに分類
    target_class, bins_rna = cut_bins_class(df[target_series])
    target_class = target_class.rename(columns={"rna_conc_log": f"{target_class}_class"})
    
    df[f"{target_series}_bins"] = target_class

    return df

print(df_rnapsec.shape)

mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_rnapsec = df_rnapsec[df_rnapsec[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
#X, y, groups
if ignore_features == False:  #省くカラムがない場合   
    X = df_rnapsec.drop([target_label, group_label, "rna_sequence", "protein_sequence", "aa_rna_label", "protein_name"], axis = "columns").reset_index(drop = True).values
else:
    X = df_rnapsec.drop([target_label, group_label, ignore_features], axis = "columns").reset_index(drop = True).values
y = df_rnapsec[target_label].values


os.makedirs(f"./result_rnapsec_pred_ex/", exist_ok = True)

df_rnapsec = log_cut_bins(df_rnapsec)
df_rnapsec_liquid = df_rnapsec[df_rnapsec.mor_label == 1]
#クラス分け
chin_ph = df_rnapsec_liquid.pH
chin_ph_bins, bins = cut_ph_class(chin_ph)
df_rnapsec_liquid["ph_bins"] = chin_ph_bins
chin_temp = df_rnapsec_liquid.temp
chin_temp_bins, bins = cut_bins_class(chin_temp)
df_rnapsec_liquid["temp_bins"] = chin_temp_bins
df_rnapsec_liquid = cut_is_bins(df_rnapsec_liquid, "ionic_strength")
#特徴量と説明変数に分ける
y_rnapsec_liquid = df_rnapsec_liquid.loc[:, ["ph_bins", "temp_bins", "ionic_strength_bins", "protein_conc_log_class", "rna_conc_log_class"]]
X_rnapsec_liquid = df_rnapsec_liquid.drop(["ph_bins", "temp_bins", "protein_conc_log_class", "rna_conc_log_class","ionic_strength_bins", "ionic_strength", 
                                     'rna_conc_log', 'protein_conc_log', 'pH', 'temp', 'mor_label', group_label,  'rna_sequence',"protein_name"
       ,'protein_sequence', 'aa_rna_label'], axis = "columns")
groups = df_rnapsec_liquid[group_label].values
#%%
# X_rnapsec_liquid.columns
# #%%
# #特徴量と説明変数に分ける
# y_rnapsec_liquid = df_rnapsec_liquid.loc[:, ["pH_bins", "temp_bins", "ionic_strength_bins", "protein_conc_log_bins", "rna_conc_log_bins"]]
# X_rnapsec_liquid = df_rnapsec_liquid.drop(["pH_bins", "temp_bins", "protein_conc_log_bins", "rna_conc_log_bins","ionic_strength_bins", "ionic_strength", 'rna_conc_log', 'protein_conc_log', 'pH', 'temp', 'mor_label', 'protein_sequence', 'rna_sequence', 'protein_name', 'aa_label','aa_rna_label' , "rnapsec_all_col_idx"], axis = "columns")
# groups = df_rnapsec_liquid[group_label].values

#%%
#モデル
def chain_classifier_kfold (X, y, fold = 10, groups = groups):  

    all_idx = list()
    models=[]
    y_probs={}
    y_tests={}
    y_preds = {}
    for i in range(0,10):
        y_probs[i]={}
        y_tests[i]={}
        y_preds[i] = {}
    models={}

    kf = GroupKFold(n_splits = fold) 
    for ((train_index, test_index), x) in zip(kf.split(X, y, groups = groups), range(0,fold)):
            all_idx.extend(test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf=lgb.LGBMClassifier(
                feature_pre_filter= False,lambda_l2= 9.5,num_leaves= 9,feature_fraction= 0.984,bagging_fraction= 1.0,bagging_freq= 0,min_child_samples= 5
            )
                
            lgb_c = ClassifierChain(clf, order=[0, 1, 2, 3, 4])
            lgb_c.fit(X_train, y_train)

            y_pred = lgb_c.predict(X_test)
            y_preds[x] = y_pred  

            X_ = lgb_c._validate_data(X_test, accept_sparse=True, reset=False)
            Y_pred_chain = np.zeros((X_.shape[0], len(lgb_c.estimators_)))
            
            models[x]=lgb_c
            
            for chain_idx, estimator in enumerate(lgb_c.estimators_):
                previous_predictions = Y_pred_chain[:, :chain_idx]
                if sp.issparse(X):
                    if chain_idx == 0:
                        X_aug = X_
                        a=y_probs[x]
                        a[chain_idx]=estimator.predict_proba(X_aug)
                        b=y_tests[x].iloc[:,chain_idx]
                        b[chain_idx]=y_test
                        
                    else:
                        X_aug = sp.hstack((X_, previous_predictions))
                        
                        a=y_probs[x]
                        a[chain_idx]=estimator.predict_proba(X_aug)
                        b=y_tests[x]
                        b[chain_idx]=y_test.iloc[:,chain_idx]
        
                else:
                    X_aug = np.hstack((X_, previous_predictions))
                    a=y_probs[x]
                    a[chain_idx]=estimator.predict_proba(X_aug)
                    b=y_tests[x]
                    b[chain_idx]=y_test.iloc[:,chain_idx]
    return y_tests, y_probs, y_preds, models
tests_rnapsec, probs_rnapsec, preds_rnapsec, models_rnapsec = chain_classifier_kfold(X = X_rnapsec_liquid, y = y_rnapsec_liquid, fold = 10)


def plot_heatmap_list(cm,title, ticklabel):
    plt.rcParams["xtick.major.pad"] = 10
    plt.rcParams["ytick.major.pad"] = 10

    fig_cm = plt.figure(figsize = (10,10), facecolor="white")
    h=sns.heatmap(cm, annot=True,cmap="Blues", fmt="d", annot_kws={'size':50}, linecolor="black", linewidths=1.0, cbar = False)
    sns.set(font_scale=3)
    # h.set_xticklabels(ticklabel, fontsize=26)
    # h.set_yticklabels(ticklabel, fontsize=26)
    
    plt.xticks(rotation = 0, fontsize = 32)
    plt.yticks(rotation = 0, fontsize = 32)
    
    plt.title(title, fontsize=50, pad=25)
    plt.xlabel("Prediction", fontsize=40, labelpad=20)
    plt.ylabel("Experiment", fontsize=40, labelpad=20)
    plt.tight_layout()
    plt.show
    return fig_cm

def reshape_tests_preds(preds, tests):
    preds_chain0 = list()
    preds_chain1 = list()
    preds_chain2 = list()
    preds_chain3 = list()
    preds_chain4 = list()
    tests_chain0 = list()
    tests_chain1 = list()
    tests_chain2 = list()
    tests_chain3 = list()
    tests_chain4 = list()
    for i in range(0, 10):
        df_preds = pd.DataFrame(preds[i])
        df_tests = pd.DataFrame(tests[i])
        for (x, m) in zip(df_preds.index, df_tests.index):
            preds_chain0.append(df_preds[0].loc[x])
            preds_chain1.append(df_preds[1].loc[x])
            preds_chain2.append(df_preds[2].loc[x])
            preds_chain3.append(df_preds[3].loc[x])
            preds_chain4.append(df_preds[4].loc[x])
            tests_chain0.append(df_tests[0].loc[m])
            tests_chain1.append(df_tests[1].loc[m])
            tests_chain2.append(df_tests[2].loc[m])
            tests_chain3.append(df_tests[3].loc[m])
            tests_chain4.append(df_tests[4].loc[m])
    tests_list = [tests_chain0, tests_chain1, tests_chain2, tests_chain3, tests_chain3]
    preds_list = [preds_chain0, preds_chain1, preds_chain2, preds_chain3, preds_chain4]
    return tests_list, preds_list 
#出力結果の整形
tests_list_rnapsec, preds_list_rnapsec = reshape_tests_preds(preds_rnapsec, tests_rnapsec)
def make_cm(chain_no, df, preds_list, tests_list, ds_name = None):

    cms = list()
    for i in range(0, chain_no):
        cm = confusion_matrix(tests_list[i], preds_list[i], )
        cms.append(cm)
    protein_bins, rna_bins, is_bins = range(5), range(5), range(5)
    ph_bins = ["0-7", "7-8", "8-14"]
    temp_bins = ["0-25", "25-38", "38-"]
    bins_list = [ph_bins, temp_bins, is_bins, protein_bins, rna_bins]
    titles = ["pH", "Temperature", "Ionic strength", "Protein concentration", "RNA concentration",]
    for (cm, title, bin_name) in zip(cms, titles, bins_list):
        fig = plot_heatmap_list(cm, f"{title}", bin_name)
        fig_title = title.replace(" ", "_")
        fig.savefig(f"./{fig_title}.png")
def label_name(df0):
    protein_bins=[f"{np.nanpercentile(df0.protein_conc_log.unique(), 0):.3f}-{np.nanpercentile(df0.protein_conc_log.unique(), 20):.3f}",
            f"{np.nanpercentile(df0.protein_conc_log.unique(), 20):.3f}-{np.nanpercentile(df0.protein_conc_log.unique(), 40):.3f}",
            f"{np.nanpercentile(df0.protein_conc_log.unique(), 40):.3f}-{np.nanpercentile(df0.protein_conc_log.unique(), 60):.3f}",
            f"{np.nanpercentile(df0.protein_conc_log.unique(), 60):.3f}-{np.nanpercentile(df0.protein_conc_log.unique(), 80):.3f}", 
            f"{np.nanpercentile(df0.protein_conc_log.unique(), 80):.3f}-{np.nanpercentile(df0.protein_conc_log.unique(), 100):.3f}"
            ]
    rna_bins=[f"{np.nanpercentile(df0.rna_conc_log.unique(), 0):.3f}-{np.nanpercentile(df0.rna_conc_log.unique(), 20):.3f}",
            f"{np.nanpercentile(df0.rna_conc_log.unique(), 20):.3f}-{np.nanpercentile(df0.rna_conc_log.unique(), 40):.3f}",
            f"{np.nanpercentile(df0.rna_conc_log.unique(), 40):.3f}-{np.nanpercentile(df0.rna_conc_log.unique(), 60):.3f}",
            f"{np.nanpercentile(df0.rna_conc_log.unique(), 60):.3f}-{np.nanpercentile(df0.rna_conc_log.unique(), 80):.3f}", 
            f"{np.nanpercentile(df0.rna_conc_log.unique(), 80):.3f}-{np.nanpercentile(df0.rna_conc_log.unique(), 100):.3f}"
            ]
    return protein_bins, rna_bins

#混同行列作成
protein_bins_rnapsec, rna_bins_rnapsec = label_name(df_rnapsec_liquid)
make_cm(5, df_rnapsec_liquid, preds_list_rnapsec, tests_list_rnapsec, ds_name = "chin")

# %%
#############
#accuracyの計算
def make_acc(chain_no,  preds_list, tests_list, ds_name = "A"):
    accs = list()
    for i in range(0, chain_no):
        acc = accuracy_score (preds_list[i], tests_list[i])
        accs.append(acc)
    df_acc = pd.DataFrame(accs)
    df_acc = df_acc.rename(columns = {0: ds_name})
    return df_acc

df_acc_rnapsec = make_acc(5,  preds_list_rnapsec, tests_list_rnapsec, ds_name = "chin")
df_acc_rnapsec.index = ["ph", "temp","ionic_strength", "protein_conc", "rna_conc"]
print(df_acc_rnapsec)
df_acc_rnapsec.to_csv(f"./acc_{group_label}_0413.csv")

#macro ROCの計算 (3クラス)
def macro_auc_chain3(test_no, chain_no, y_tests, y_probs):
    n_classes=3
    M_y_score = []
    classes=range(n_classes)
    y_test_b=[]
    for x in range(0,10):
        y_test_b.append(label_binarize(y_tests[x][test_no].values, classes=[0,1,2,]))
        for i in range(0, y_probs[x][0].shape[0]):
            M_y_score.append([y_probs[x][chain_no][i,0],y_probs[x][chain_no][i,1], y_probs[x][chain_no][i,2]])
    y_test_c=[]
    for x in range(0,10):
        for i in range(0, y_test_b[x].shape[0]):
            y_test_c.append([y_test_b[x][i,0], y_test_b[x][i,1], y_test_b[x][i,2],])
    df_y_test_c=pd.DataFrame(y_test_c)
    auc_m1 = roc_auc_score(y_test_c, M_y_score, multi_class="ovo")
    M_y_score=np.array(M_y_score)
    M_fpr = dict()
    M_tpr = dict()
    M_roc_auc = dict()
    n_class = 5
    from sklearn.metrics import roc_curve, auc
    for i in range(n_classes):
        M_fpr[i], M_tpr[i], _ = roc_curve(df_y_test_c.loc[:, i], M_y_score[:, i])
        M_roc_auc[i] = auc(M_fpr[i], M_tpr[i])
        
    M_all_fpr = np.unique(np.concatenate([M_fpr[i] for i in range(n_classes)]))
    M_mean_tpr = np.zeros_like(M_all_fpr)
    for i in range(n_classes):
        M_mean_tpr += np.interp(M_all_fpr, M_fpr[i], M_tpr[i])

    M_mean_tpr /= n_classes
    M_fpr["macro"] = M_all_fpr
    M_tpr["macro"] = M_mean_tpr
    M_roc_auc["macro"] = auc(M_fpr["macro"], M_tpr["macro"])
    return  M_fpr, M_tpr, M_roc_auc["macro"]
#macro rocの計算 (5クラス)
def macro_auc_chain(test_no, chain_no, y_tests, y_probs):
    n_classes=5
    M_y_score = []
    y_test_b=[]
    for x in range(0,10):
        y_test_b.append(label_binarize(y_tests[x][test_no].values, classes=[0,1,2,3,4]))
        for i in range(0, y_probs[x][0].shape[0]):
            M_y_score.append([y_probs[x][chain_no][i,0],y_probs[x][chain_no][i,1], y_probs[x][chain_no][i,2], y_probs[x][chain_no][i,3], y_probs[x][chain_no][i,4]])
    y_test_c=[]
    for x in range(0,10):
        for i in range(0, y_test_b[x].shape[0]):
            y_test_c.append([y_test_b[x][i,0], y_test_b[x][i,1], y_test_b[x][i,2], y_test_b[x][i,3], y_test_b[x][i,4]])
    df_y_test_c=pd.DataFrame(y_test_c)
    M_y_score=np.array(M_y_score)
    M_fpr = dict()
    M_tpr = dict()
    M_roc_auc = dict()
    from sklearn.metrics import roc_curve, auc
    for i in range(n_classes):
        M_fpr[i], M_tpr[i], _ = roc_curve(df_y_test_c.loc[:, i], M_y_score[:, i])
        M_roc_auc[i] = auc(M_fpr[i], M_tpr[i])
        
    M_all_fpr = np.unique(np.concatenate([M_fpr[i] for i in range(n_classes)]))
    M_mean_tpr = np.zeros_like(M_all_fpr)
    for i in range(n_classes):
        M_mean_tpr += np.interp(M_all_fpr, M_fpr[i], M_tpr[i])

    M_mean_tpr /= n_classes
    M_fpr["macro"] = M_all_fpr
    M_tpr["macro"] = M_mean_tpr
    M_roc_auc["macro"] = auc(M_fpr["macro"], M_tpr["macro"])
    return  M_fpr, M_tpr, M_roc_auc["macro"]

#実験条件ごとにROC-AUCなどを計算
chin0_fpr, chin0_tpr, chin0_roc_auc = macro_auc_chain3(0, 0,  tests_rnapsec,probs_rnapsec, )
chin1_fpr, chin1_tpr, chin1_roc_auc = macro_auc_chain3(1, 1,  tests_rnapsec,probs_rnapsec, )
chin2_fpr, chin2_tpr, chin2_roc_auc = macro_auc_chain(2, 2,  tests_rnapsec,probs_rnapsec, )
chin3_fpr, chin3_tpr, chin3_roc_auc = macro_auc_chain(3, 3,  tests_rnapsec,probs_rnapsec, )
chin4_fpr, chin4_tpr, chin4_roc_auc = macro_auc_chain(4, 4,  tests_rnapsec,probs_rnapsec, )

#作図
fprs = [chin0_fpr, chin1_fpr, chin2_fpr, chin3_fpr, chin4_fpr]
tprs = [chin0_tpr, chin1_tpr, chin2_tpr, chin3_tpr, chin4_tpr]
roc_aucs = [chin0_roc_auc, chin1_roc_auc, chin2_roc_auc, chin3_roc_auc, chin4_roc_auc]

def plot_roc(fprs, tprs, roc_aucs, labels, title):
    lw=1
    # colors = [cm.gist_ncar(190), cm.gist_ncar(30), cm.gist_ncar(200),cm.gist_ncar(60),cm.gist_ncar(100),cm.gist_ncar(150)]
    # sns.color_palette(colors)
    # sns.set_palette(colors, desat=1.0)

    cmap =  plt.get_cmap("Set1")
    fig_roc = plt.figure(figsize=(10, 10), dpi = 100, facecolor="white")
    plt.rcParams["xtick.major.pad"] = 10
    plt.rcParams["ytick.major.pad"] = 10
    plt.style.use("classic")
    for (fpr, tpr, roc_auc, label, i) in zip(fprs, tprs, roc_aucs, labels, range(0, len(fprs))):
        plt.plot(fpr["macro"], tpr["macro"],
                label=f"{label} (AUC = {roc_auc:.2f})",
                # color=f"C{i}",  
                color = cmap(i),
                linestyle='-', linewidth=3)
        
        plt.plot([0,1,0],[0,1,1], linestyle='--', color = "black", lw = 1.2)
    plt.xlabel("True Positive Rate", fontsize = 32, labelpad=20)
    plt.ylabel("False Positive Rate", fontsize = 32, labelpad=20)
    plt.legend(loc = "lower right", fontsize = 24)
    plt.xticks(fontsize = 26)
    plt.yticks(fontsize = 26)
    plt.grid()
    plt.title(f"{title}", fontsize = 35, pad = 30)
    plt.tight_layout()
    plt.show()
    fig_roc.savefig(f"./macro_roc_llps_ex.png", bbox_inches='tight')
    return fig_roc

fig = plot_roc(fprs, tprs, roc_aucs, labels = ["pH", "Temp.","Ionic strength", "Protein conc.", "RNA conc.",], title = " ROC curve for group 10-fold CV")

# %%

######
# files =f"../data/{file_name}.csv"
# print(files)
# #クラスの番号→範囲
# def def_class(y_series_1, percentiles = [0, 20, 40, 60, 80]):
#     bins = []
#     for i in percentiles:
#         bins.append(np.percentile(y_series_1.unique(), i))
#     bins.append(np.inf)
#     print(bins)
#     return bins
# def cut_to_bins(y_series_1, bins, target_col):
#     df_bins=pd.DataFrame(y_series_1)
#     bins_names = range(len(bins)-1)
#     df_bins=pd.cut(df_bins.iloc[:,0], bins ,labels=bins_names,right=False)
#     df_bins=df_bins.to_frame()
#     df_bins=df_bins.rename(columns={df_bins.columns[0]: f"{target_col}_bins"})
#     return df_bins[f"{target_col}_bins"]

# def value_to_class(ex_series, target_col, percentile, df_input):
#     bins = def_class(df_liquid[target_col], percentile, )
#     series_bins = cut_to_bins(ex_series, bins, target_col)
#     df_input[f"{target_col}_bins"] = series_bins
#     return df_input, bins
# pH_bins = [0, 7, 8]
# temp_bins = [0, 25, 38,]
# bins_class = [range(5), range(5), range(5), pH_bins, temp_bins]
# files =f"../data/{file_name}.csv"
# print(files)
# df = pd.read_csv(files, index_col=False) 
# df = df[df[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)
# df_liquid = df[df.mor_label == 1]
# target_col = [  'pH', 'temp','protein_conc_log','ionic_strength',"rna_conc_log", ]
# bins_dict = {}
# for n, col in enumerate(target_col):
#     print(n)
#     df, bins = value_to_class(df[col].values, col, bins_class[n], df)
#     bins_dict[col] = bins
# #%%
# df.columns
# #%%
# df_rnapsec_liquid = df[df.mor_label == 1]
########