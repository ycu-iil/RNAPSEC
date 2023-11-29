#%%
"ROC, metrics"
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np
import warnings
warnings.simplefilter("ignore")

#交差検証とROC曲線、valデータ付き (分割方法はconfig_.yamlで指定）、パラメーター調整なし
with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])
min_num = args["min_num"]
files =f"../data/{file_name}.csv"
df_rnapsec = pd.read_csv(files, index_col=False) 
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_rnapsec = df_rnapsec[df_rnapsec[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)
max_num= df_rnapsec.shape[0]

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize = (15, 15))
models = ["AdaBoost","LightGBM", "Logistic Regression", "GaussianNB","KNN", "Random Forest",] 
total_score =[]
scores = []
for n, model_name in enumerate(["AdaBoostClassifier", "LGBMClassifier","LogisticRegression", "GaussianNB","KNeighborsClassifier",  "RandomForestClassifier",]):# 
    model_score = []
    kf_state_aucs = []
    for kf_state in range(10, 101, 10):
        result_path = f"result_repeated_StratifiedGroupKFold/{model_name}/kf_state_{kf_state}"
        i_list=range(10)
        total=0    
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        mean_tprs = []
        
        for i in i_list:
            test= pd.read_csv(f"{result_path}/pred_cutdata_val_{min_num}_{max_num}_fold{i}.csv")
            total=total+test['liquid']
            y_pred=test['liquid']
            y_tests=test['actual'].astype("int")
            df_score = pd.DataFrame()
            df_score.loc[i,"fold"] = i
            df_score["model"] = model_name
            df_score["kf_state"] = kf_state
            df_score["recall"] = metrics.recall_score(y_tests, test["preds"])
            df_score["precision"] = metrics.precision_score(y_tests, test["preds"])
            df_score["accuracy"]= metrics.accuracy_score(y_tests, test["preds"])
            f1_score = metrics.f1_score(y_tests, test["preds"])
            df_score["f1_score"] = f1_score
            
            if y_tests.unique().shape[0]>1:
                def y_test(x):
                    return str(x).replace('liquid','1').replace('solute','0')
                y_test=y_tests.map(y_test)
        #         print(y_test)
                #AUCを求める
                fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred,drop_intermediate=False,pos_label='1')
                auc = metrics.auc(fpr,tpr)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(auc)
                df_score["roc_auc"] = auc
                precision_avg, recall_avg, thresholds = metrics.precision_recall_curve(y_test,y_pred,pos_label=1)
                pr_auc = metrics.auc(recall_avg, precision_avg)
                df_score["pr_auc"] = pr_auc
                
            else:
                df_score["roc_auc"] = np.nan
                df_score["pr_auc"] =np.nan    
            scores.append(df_score)
        mean_tpr = np.mean(tprs, axis=0) #各シードでの平均
        mean_tprs.append(mean_tpr)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        kf_state_aucs.append(mean_auc)
        ax.legend(loc="lower right")
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="ROC curve for LOGO CV")
        print(kf_state)
        print(mean_auc)
    mean_fpr = np.linspace(0, 1, 100) #全シードの平均
    mean_tpr_model = np.mean(mean_tprs, axis=0)
    mean_tpr_model[-1] = 1.0
    mean_total_auc = metrics.auc(mean_fpr, mean_tpr_model)
    std_auc = np.std(kf_state_aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr_model + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr_model - std_tpr, 0)
    ax.plot(mean_fpr, mean_tpr_model, color = f"C{n}", #color=cmap(n)
                label=f'{models[n]} (%0.2f$\pm$%.2f)' % (mean_total_auc, std_auc), 
                lw=5, alpha=1)

plt.plot(np.linspace(1, 0, len(fpr)),np.linspace(1, 0, len(fpr)), linestyle = '--', color = "black")
plt.plot([0,0,1],[0,1,1], linestyle='--', alpha = 0.6, color = "black")
plt.legend(fontsize = 28) 
plt.title(f"ROC curves of Repeated SG10CV", fontsize = 40, pad = 30  )
plt.xlabel('False Positive Rate', fontsize = 40, labelpad=15)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.ylabel('True Positive Rate', fontsize = 40, labelpad=15)
plt.grid(True)
fig.savefig("result_repeated_StratifiedGroupKFold/avg_roc_curve.png")
plt.show()
plt.close()
