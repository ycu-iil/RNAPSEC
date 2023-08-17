import numpy as np
import pandas as pd
import yaml
import pickle
import matplotlib.pyplot as plt

with open("./config_.yaml",'r')as f:
    args = yaml.safe_load(f)
target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
mor_class = np.array(args["mor_class"])
min_num = args["min_num"]
file_name = args["file_name"]


mor = [["solute", "liquid", "solid"][i] for i in mor_class] #予測対象の形態の名称リスト
df = pd.read_csv( f"../data/{file_name}.csv", index_col=False)
df = df[df[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)
max_num = df.shape[0]

# for model in [ "LGBMClassifier", "AdaBoostClassifier"]:
model = "AdaBoostClassifier"
result_path = f"./result_logocv/{model}"
#カラム名変更
fis = {}
# features =  df.columns.drop([target_label, group_label, "rna_sequence", "aa", "protein_sequence", "aa_rna", "aa_rna_label", "aa_label", "group_label"])
features =  df.columns.drop([target_label, group_label, "aa_rna_label", "rnapsec_all_col_idx", "protein_sequence", "rna_sequence", "protein_name"])
print(features.shape)
features = features.str.replace("_", " ")
features = features.str.replace("rna rna", "rna")
features = features.str.replace("protein", "(protein)")
features = features.str.replace("rna", "(RNA)")
features = features.str.replace(" conc", " conc.")
features = features.str.replace("log", "(log)")
features = features.str.replace("temp", "Temperature")
features = features.str.replace("ionic strength", "Ionic strength")
for i in range(df[group_label].unique().shape[0]):
    with open(f"{result_path}/model_list_{file_name}.pickle", 'rb') as web:
        clf = pickle.load(web)
    fis[i] = clf[i].feature_importances_
df_fi=pd.DataFrame(fis, index=features)
df_fi["avg"]=df_fi.mean(axis="columns")
# df_fi.to_csv(f"{result_path}/feature_importances_{file_name}_mor{len(mor_class)}_{fold_strategy}.csv", index = False)
#作図
def plt_fi(df_fi, db_name):
    fig_fi = plt.figure()
    df_fi["avg"].sort_values( ascending=True).tail(10).plot.barh(figsize=(10,8),label="methods = split", fontsize=20)
    plt.xlabel("Feature importance",  fontsize=20)
    plt.ylabel("feature",  fontsize=20)
    plt.legend(loc="lower right", fontsize=18)
    plt.grid()
    plt.title(f"{model}\n fold strategy = LOGOCV",  fontsize=24, pad = 20)
    # fig_fi.savefig(f"{result_path}/feature_importances_{file_name}_mor{len(mor_class)}_{fold_strategy}.png", bbox_inches='tight')
    return 
plt_fi(df_fi, file_name)