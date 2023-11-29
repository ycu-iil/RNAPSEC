import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import ClassifierChain
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
# 関数系
# value→class function
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
#溶質濃度
def cut_solute_conc_bins(df):
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
def cut_ph_bins(y_series_1):
    df_ph=pd.DataFrame(y_series_1)
    bins = [0, 7, 8, 1000]
    bins_names=[0, 1, 2]
    df_ph_class=pd.cut(df_ph.iloc[:,0],bins ,labels=bins_names,right=False)
    df_ph_class=df_ph_class.to_frame()
    df_ph_class=df_ph_class.rename(columns={f"{y_series_1}": f"{y_series_1}_class"})
    return  df_ph_class, bins

def cut_is_bins(df, target_series):
    #log_concを5クラスに分類
    target_class, bins_is = cut_bins_class(df[target_series])
    target_class = target_class.rename(columns={"rna_conc_log": f"{target_class}_class"})
    df[f"{target_series}_bins"] = target_class
    return df, bins_is
#温度→低音、室温、体温、高温の4クラス分類
def cut_temp_bins(y_series_1, bins_list = [0, 25, 30, 38, 1000]):
    df_temp=pd.DataFrame(y_series_1)
    bins = bins_list
    bins_names=[0, 1, 2, 3]
    df_temp_class=pd.cut(df_temp.iloc[:,0],bins ,labels=bins_names,right=False)
    df_temp_class=df_temp_class.to_frame()
    df_temp_class=df_temp_class.rename(columns={f"{y_series_1}": f"{y_series_1}_bins"})
    return  df_temp_class, bins

def chain_classifier_kfold (X, y, groups, fold = 10, ):  
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
            clf=AdaBoostClassifier(random_state = 42)    
            chain = ClassifierChain(clf, order=[0, 1, 3, 4,2])
            chain.fit(X_train, y_train)

            y_pred = chain.predict(X_test)
            y_preds[x] = y_pred  
            X_ = chain._validate_data(X_test, accept_sparse=True, reset=False)
            Y_pred_chain = np.zeros((X_.shape[0], len(chain.estimators_)))
            models[x]=chain
            
            for chain_idx, estimator in enumerate(chain.estimators_):
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

def reshape_tests_preds(preds, tests): #モデルの計算結果を整形
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

def label_name(df0): #混同行列の軸ラベルの名称
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
def plot_heatmap_list(cm,title): #heatmap作成
    plt.rcParams["xtick.major.pad"] = 10
    plt.rcParams["ytick.major.pad"] = 10

    fig_cm = plt.figure(figsize = (10,10), facecolor="white")
    sns.heatmap(cm, annot=True,cmap="Blues", fmt="d", annot_kws={'size':50}, linecolor="black", linewidths=1.0, cbar = False)
    sns.set(font_scale=3)
    plt.xticks(rotation = 0, fontsize = 32)
    plt.yticks(rotation = 0, fontsize = 32)
    plt.title(title, fontsize=50, pad=25)
    plt.xlabel("Prediction", fontsize=40, labelpad=20)
    plt.ylabel("Experiment", fontsize=40, labelpad=20)
    plt.tight_layout()
    plt.show
    return fig_cm
def make_cm(chain_no, preds_list, tests_list): #混同行列作成→図として表示
    cms = list()
    for i in range(0, chain_no):
        cm = confusion_matrix(tests_list[i], preds_list[i], )
        cms.append(cm)
    protein_bins, rna_bins, is_bins = range(5), range(5), range(5)
    ph_bins = ["0-7", "7-8", "8-14"]
    temp_bins = ["0-20", "20-30", "30-40", "40-"]
    bins_list = [ph_bins, temp_bins,  protein_bins, rna_bins,is_bins,]
    titles = ["pH", "Temperature", "Protein concentration", "RNA concentration","Ionic strength", ]
    for (cm, title) in zip(cms, titles):
        fig = plot_heatmap_list(cm, f"{title}")
        fig_title = title.replace(" ", "_")
        fig.savefig(f"./{fig_title}.png")
    return
#各条件の精度計算
def calc_acc(chain_no,  preds_list, tests_list, ds_name = "A"):
    accs = list()
    for i in range(0, chain_no):
        acc = accuracy_score (preds_list[i], tests_list[i])
        accs.append(acc)
    df_acc = pd.DataFrame(accs)
    df_acc = df_acc.rename(columns = {0: ds_name})
    return df_acc
# macro ROC-AUCの計算
def macro_auc_chain(test_no, chain_no, y_tests, y_probs, y_rnapsec_liquid ):
    n_classes = len(y_rnapsec_liquid.iloc[:, chain_no].unique())
    y_test_b=[]
    for x in range(len(y_tests)):
        df_binary = pd.DataFrame(label_binarize(y_tests[x][test_no].values, classes=range(n_classes)))
        y_test_b.append(df_binary)
    df_y_test_c = pd.concat(y_test_b).reset_index(drop = True)
    tst = []
    for i in range(len(y_tests)):
        tst.append(pd.DataFrame(y_probs[i][chain_no]))
    df_x_test = pd.concat(tst).reset_index(drop = True)
    M_fpr = dict()
    M_tpr = dict()
    M_roc_auc = dict()
    from sklearn.metrics import roc_curve, auc
    for i in range(n_classes):
        M_fpr[i], M_tpr[i], _ = roc_curve(df_y_test_c.loc[:, i], df_x_test.loc[:, i])
        M_roc_auc[i] = auc(M_fpr[i], M_tpr[i]) #各
    M_all_fpr = np.unique(np.concatenate([M_fpr[i] for i in range(n_classes)]))
    M_mean_tpr = np.zeros_like(M_all_fpr)
    for i in range(n_classes):
        M_mean_tpr += np.interp(M_all_fpr, M_fpr[i], M_tpr[i])
    M_mean_tpr /= n_classes
    M_fpr["macro"] = M_all_fpr
    M_tpr["macro"] = M_mean_tpr
    M_roc_auc["macro"] = auc(M_fpr["macro"], M_tpr["macro"])
    return M_fpr, M_tpr, M_roc_auc["macro"]
#ROC曲線をプロット
def plot_roc(fprs, tprs, roc_aucs, labels, title):
    cmap =  plt.get_cmap("Set1")
    fig_roc = plt.figure(figsize=(10, 10), dpi = 100, facecolor="white")
    plt.rcParams["xtick.major.pad"] = 10
    plt.rcParams["ytick.major.pad"] = 10
    plt.style.use("classic")
    for (fpr, tpr, roc_auc, label, i) in zip(fprs, tprs, roc_aucs, labels, range(0, len(fprs))):
        plt.plot(fpr["macro"], tpr["macro"],
                label=f"{label} ({roc_auc:.2f})",
                color = cmap(i),
                linestyle='-', linewidth=3)
        
        plt.plot([0,1,0],[0,1,1], linestyle='--', color = "black", lw = 1.2)
    plt.xlabel("True Positive Rate", fontsize = 32, labelpad=20)
    plt.ylabel("False Positive Rate", fontsize = 32, labelpad=20)
    plt.legend(loc = "lower right", fontsize = 20).get_frame().set_alpha(0.7)
    plt.xticks(fontsize = 26)
    plt.yticks(fontsize = 26)
    plt.grid()
    plt.title(f"{title}", fontsize = 35, pad = 30)
    plt.tight_layout()
    plt.show()
    fig_roc.savefig(f"./macro_roc_llps_ex.png", bbox_inches='tight')
    return fig_roc

def main():
    with open("./config_.yaml",'r')as f:
        args = yaml.safe_load(f)

    target_label = args["target_label"]
    group_label = args["group_label"]
    file_name = args["file_name"]
    mor_class = np.array(args["mor_class"])
    # read file
    files =f"../data/{file_name}.csv"
    print(files)
    df_rnapsec = pd.read_csv(files, index_col=False) 
    # filtering by liquid data
    mor_class = [1]
    df_rnapsec_liquid = df_rnapsec[df_rnapsec[target_label].isin(mor_class)].reset_index(drop=True)
    #value → class
    df_rnapsec_liquid = cut_solute_conc_bins(df_rnapsec_liquid)
    rnapsec_ph = df_rnapsec_liquid.pH
    rnapsec_ph_bins, bins = cut_ph_bins(rnapsec_ph)
    df_rnapsec_liquid["ph_bins"] = rnapsec_ph_bins
    rnapsec_temp = df_rnapsec_liquid.temp
    rnapsec_temp_bins, bins = cut_temp_bins(rnapsec_temp)
    df_rnapsec_liquid["temp_bins"] = rnapsec_temp_bins
    df_rnapsec_liquid, is_bins = cut_is_bins(df_rnapsec_liquid, "ionic_strength")
    # defined X, y, groups for ML models
    y_rnapsec_liquid = df_rnapsec_liquid.loc[:, ["ph_bins", "temp_bins", "ionic_strength_bins", "protein_conc_log_class", "rna_conc_log_class"]]
    X_rnapsec_liquid = df_rnapsec_liquid.drop(['rnapsec_all_col_idx', "ph_bins", "temp_bins", "protein_conc_log_class", "rna_conc_log_class",
                                            "ionic_strength_bins", "ionic_strength", 
                                            'rna_conc_log', 'protein_conc_log', 'pH', 'temp', 'mor_label', group_label,  'rna_sequence',"protein_name",
                                            'protein_sequence', 'aa_rna_label'], axis = "columns")
    groups = df_rnapsec_liquid[group_label].values
    # cross-validatation
    tests_rnapsec, probs_rnapsec, preds_rnapsec, models_rnapsec = chain_classifier_kfold(X = X_rnapsec_liquid, y = y_rnapsec_liquid,groups=groups, fold = 10)
    # 出力結果の整形
    tests_list_rnapsec, preds_list_rnapsec = reshape_tests_preds(preds_rnapsec, tests_rnapsec)
    #混同行列作成
    protein_bins_rnapsec, rna_bins_rnapsec = label_name(df_rnapsec_liquid) #溶質濃度の軸ラベルの名前付け
    print("value for each class of the protein concentration", protein_bins_rnapsec)
    print("value for each class of the RNA concentration",rna_bins_rnapsec)
    make_cm(5, preds_list_rnapsec, tests_list_rnapsec, )
    # 各条件の予測精度計算→csv出力
    df_acc_rnapsec = calc_acc(5,  preds_list_rnapsec, tests_list_rnapsec)
    df_acc_rnapsec.index = ["ph", "temp", "protein_conc", "rna_conc","ionic_strength",]
    print(df_acc_rnapsec)
    df_acc_rnapsec.to_csv(f"./acc_{group_label}_1127.csv")
    #macro ROCの計算→ROC曲線作成
    rnapsec0_fpr, rnapsec0_tpr, rnapsec0_roc_auc = macro_auc_chain(0, 0,  tests_rnapsec,probs_rnapsec,y_rnapsec_liquid )
    rnapsec1_fpr, rnapsec1_tpr, rnapsec1_roc_auc = macro_auc_chain(1, 1,  tests_rnapsec,probs_rnapsec, y_rnapsec_liquid)
    rnapsec2_fpr, rnapsec2_tpr, rnapsec2_roc_auc = macro_auc_chain(2, 2,  tests_rnapsec,probs_rnapsec, y_rnapsec_liquid)
    rnapsec3_fpr, rnapsec3_tpr, rnapsec3_roc_auc = macro_auc_chain(3, 3,  tests_rnapsec,probs_rnapsec, y_rnapsec_liquid)
    rnapsec4_fpr, rnapsec4_tpr, rnapsec4_roc_auc = macro_auc_chain(4, 4,  tests_rnapsec,probs_rnapsec, y_rnapsec_liquid)
    fprs = [rnapsec0_fpr, rnapsec1_fpr, rnapsec2_fpr, rnapsec3_fpr, rnapsec4_fpr]
    tprs = [rnapsec0_tpr, rnapsec1_tpr, rnapsec2_tpr, rnapsec3_tpr, rnapsec4_tpr]
    roc_aucs = [rnapsec0_roc_auc, rnapsec1_roc_auc, rnapsec2_roc_auc, rnapsec3_roc_auc, rnapsec4_roc_auc]
    #ROC曲線プロット
    fig = plot_roc(fprs, tprs, roc_aucs, labels = ["pH", "Temp.","Protein conc.", "RNA conc.","Ionic strength", ], title = " Macro-averaged ROC curves of G10CV")

if __name__=="__main__":
    main()