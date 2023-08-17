#%%
#%%
import numpy as np
import pandas as pd
import yaml
import glob
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)
min_num = 0
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])
target_label = args["target_label"]
group_label = args["group_label"]
files =f"../data/{file_name}.csv"
print(files)
df = pd.read_csv(files, index_col=False) 
df = df[df[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)
max_num= df.shape[0]

data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}



models_result_list = []
model_list = [ "LGBMClassifier","AdaBoostClassifier", "RandomForestClassifier", ]
for model in model_list:
    result_list =[]
    result_path = f"./result_logocv/{model}"
    for i in range(df[group_label].unique().shape[0]):
        df_result = pd.read_csv(f'{result_path}/pred_cutdata_val_{min_num}_{max_num}_fold{i}.csv',index_col=False)
        df_result["actual"] = df_result["actual"].astype("int")
        df_result["preds"] = df_result["preds"].astype("int")
        tp = df_result[(df_result.preds == 1) & (df_result.actual == 1)]
        fp = df_result[(df_result.preds == 1) & (df_result.actual == 0)]
        tn = df_result[(df_result.preds == 0) & (df_result.actual == 0)]
        fn = df_result[(df_result.preds == 0) & (df_result.actual == 1)]
        #recall
        # recall = tp.shape[0]/(tp.shape[0] + fn.shape[0])
        recall =  metrics.recall_score(df_result.actual, df_result.preds)
        print("recall (manual): ", recall)
        print("recall (auto: metrics): ", metrics.recall_score(df_result.actual, df_result.preds))
        #macro_recall
        macro_recall = metrics.recall_score(df_result.actual, df_result.preds, average="macro")
        print("macro_recall", macro_recall)
        #precision
        # precision = tp.shape[0]/(tp.shape[0] + fp.shape[0])
        # print("precision (manual): ", precision)
        precision =  metrics.precision_score(df_result.actual, df_result.preds)
        print("precision (auto: metrics): ", metrics.precision_score(df_result.actual, df_result.preds))
        #macro_precision
        macro_precision = metrics.precision_score(df_result.actual, df_result.preds, average="macro")
        print("macro_precision", macro_precision)
        #f1-score
        # f1_score = 2* ((precision*recall)/(precision+recall))
        # print("f1_score (manual): ", f1_score)
        print("f1_score (auto: metrics): ", metrics.f1_score(df_result.actual, df_result.preds))
        f1_score = metrics.f1_score(df_result.actual, df_result.preds)
        #neg_label f1_score -> pos_label = 0
        neg_f1 = metrics.f1_score(df_result.actual, df_result.preds, pos_label = 0)
        print("neg_f1: ", neg_f1)
        #macro f1_score
        macro_f1 = metrics.f1_score(df_result.actual, df_result.preds, average="macro")
        print("macro_f1: ", macro_f1)
        #accuracy
        accuracy = (tp.shape[0]+tn.shape[0])/df_result.shape[0]
        print("accuracy (manual): ", accuracy)
        print("accuracy (auto: metrics): ", metrics.accuracy_score(df_result.actual, df_result.preds))
        #clf_report
        print(classification_report(df_result.actual, df_result.preds))
        #roc-auc
        fpr, tpr, thresholds = metrics.roc_curve(df_result.actual, df_result.liquid)
        roc_auc = metrics.auc(fpr,tpr)
        print("roc_auc", roc_auc)
        # pos_label逆転させて、roc_auc: 
        df_rev_act = pd.DataFrame(label_binarize(df_result.actual, classes = [0, 1, 2])[:, :2])
        fpr_rev, tpr_rev, thresholds = metrics.roc_curve(df_rev_act.loc[:, 1], df_result.solute)
        roc_auc_rev = metrics.auc(fpr_rev, tpr_rev)
        print("roc_auc_rev", roc_auc_rev)
        #pr-auc
        precision_avg, recall_avg, thresholds = metrics.precision_recall_curve(df_result.actual, df_result.liquid)
        pr_auc = metrics.auc(recall_avg, precision_avg)
        #cohen kappa
        kappa = metrics.cohen_kappa_score(df_result.preds, df_result.actual)
        print("kappa", kappa)
        #hinge_loss 
        hinge_loss = metrics.hinge_loss(df_result.actual, df_result.liquid)
        #MCC #https://sammi-baba.hatenablog.com/entry/2019/07/31/171213
        #　マシューズ相関係数: 日分類結果と実際結果の相関係数
        mcc = metrics.matthews_corrcoef(df_result.actual, df_result.preds)
        print("mcc: ", mcc)
        #fbeta_score
        # fbeta  = metrics.fbeta_score(df_result.actual, df_result.preds)
        # print("fbeta score:", fbeta)
        #speficity
        # specificity = (tn.shape[0]/(tn.shape[0] + fp.shape[0]))

        df_score = pd.DataFrame()
        df_score.loc[f"{model}", "roc_auc"] = roc_auc
        df_score["pr_auc"] = pr_auc
        df_score["accuracy"] = accuracy
        # df_score["specificity"] = specificity
        df_score["precision"] = precision
        df_score["recall"] =recall
        df_score["f1_score"] =f1_score
        df_score["macro_precision"] = macro_precision
        df_score["macro_recall"] = macro_recall
        df_score["macro_f1"] = macro_f1
        df_score["kappa"] = kappa
        df_score["mcc"] = mcc
        df_score["hinge_loss"] = hinge_loss
        df_score["neg_f1"] = neg_f1
        df_score["neg_roc"] = roc_auc_rev
        # df_score.to_csv(f"./result_cutdata_{file_name}/{model}/data_{min_num}_{max_num}/std_false/opt_false/cv_{fold_strategy}_valsplit_{val_fold_strategy}/es_{early_stopping_num}_test{random_state}_val_{val_fold}/{group_label}/score.csv", index = False)
        result_list.append(df_score)
    df_scores = pd.concat(result_list, axis = "index").mean(axis="index").T
    models_result_list.append(df_scores)
    plt.figure(figsize = (10,  10))
    plt.plot(recall_avg,precision_avg, label = f"{file_name} ({max_num})\n auc=%.2f"%pr_auc)
    plt.grid()
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.legend(fontsize = 26)
    plt.title(f"({model}) PR-curve {file_name} ({max_num})\n cv=LOGO CV, model={model}", fontsize = 32, pad = 20)
    plt.xlabel("Precision", fontsize = 28, labelpad = 10)
    plt.ylabel("Recall", fontsize = 28, labelpad = 10)
    plt.savefig(f"{result_path}/pr_curve.png")
    print(model)
#%%
models_result = pd.concat(models_result_list, axis="columns")
models_result.columns= model_list[:-1]
#%%
models_result.T.to_csv(f"./result_logocv/macro_score_all_models.csv")

#%%
df_result
# %%
