#%%
"ROC, metrics"
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import glob
import yaml
import numpy as np
import os

os.makedirs("./evaluateion_result/", exist_ok=True)

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
print(files)
df_rnapsec = pd.read_csv(files, index_col=False) 

mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_rnapsec = df_rnapsec[df_rnapsec[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)
max_num= df_rnapsec.shape[0]
#X, y, groups
if ignore_features == False:  #省くカラムがない場合   
    X = df_rnapsec.drop([target_label, group_label, "rna_sequence", "protein_sequence",  "aa_rna_label", "aa_label", ], axis = "columns").reset_index(drop = True).values
else:
    print(ignore_features)
    X = df_rnapsec.drop([target_label, group_label], axis = "columns").reset_index(drop = True)
    X = X.drop(ignore_features, axis = "columns").reset_index(drop = True).values
y = df_rnapsec[target_label].values
groups = df_rnapsec[group_label].values

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize = (15, 15))
models = ["lightGBM", "AdaBoost", "Random Forest", "KNN", "Logistic Regression", "GaussianNB"]
for n, modeel_name in enumerate(["LGBMClassifier", "AdaBoostClassifier", "KNeighborsClassifier", "LogisticRegression", "GaussianNB"]):
    result_path = f"result_logocv/{modeel_name}"
    i_list=df_rnapsec[group_label].unique()
    total=0    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in i_list:
        test= pd.read_csv(f"{result_path}/pred_cutdata_val_{min_num}_{max_num}_fold{i}.csv")
        total=total+test['liquid']
    # if test.preds.unique().shape[0]>=2:
        y_pred=test['liquid']
        y_tests=test['actual'].astype("int")
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
        else:
            continue
    print(modeel_name)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    ax.plot(mean_fpr, mean_tpr, color = f"C{n}", #color=cmap(n)
            label=f'{models[n]} (area = %0.2f, $\pm$ %.1f)' % (mean_auc, std_auc), 
            lw=5, alpha=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color = f"C{n}", alpha=.2,
                    )#label=r'$\pm$ 1 std. dev.'
    ax.legend(loc="lower right")
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
    title="ROC curve for LOGO CV")

plt.plot(np.linspace(1, 0, len(fpr)),np.linspace(1, 0, len(fpr)), label=f'Random (area = %.2f)'%0.5, linestyle = '--', color = "black")
plt.plot([0,0,1],[0,1,1], linestyle='--',label='Ideal (area = %.2f)'%1.0, alpha = 0.6, color = "black")
plt.legend(fontsize = 28) 
plt.title(f"ROC curve for LOGO CV", fontsize = 40, pad = 30  )
plt.xlabel('False Positive Rate', fontsize = 40, labelpad=15)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.ylabel('True Positive Rate', fontsize = 40, labelpad=15)
plt.grid(True)
plt.show()
day = datetime.date.today()
today = day.strftime("%Y%m%d")
plt.savefig(f'roc_logocv.png')

"""
#平均のAUCを求める
test= pd.read_csv(f'../result/{model}/cv_result/total1.csv')
total=total+test['liquid']
y_pred_ave=test['liquid']
y_tests=test['actual']
def y_test(x):
    return str(x).replace('liquid','1').replace('solute','0')
y_test=y_tests.map(y_test)
fpr_ave, tpr_ave, thresholds_ave = metrics.roc_curve(y_test,y_pred_ave,drop_intermediate=False,pos_label='1')
auc_ave = metrics.auc(fpr_ave,tpr_ave)
print(auc_ave)
#AUCの値のCSV作成

#平均のROC曲線を描く


#plt.plot(fpr_ave, tpr_ave, label='ROC curve (area = %.2f)'%auc_ave, color = "black",lw = 1.3 )"""

# %%
mean_tpr

# %%


y_tests
# %%
