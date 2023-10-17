
from sklearn.multioutput import ClassifierChain
import numpy as np
import lightgbm as lgb
import pandas as pd
import yaml
import pickle
import matplotlib  # <--ここを追加
matplotlib.use('Agg')  # https://python-climbing.com/runtimeerror_main_thread_is_not_in_main_loop/
from matplotlib import pyplot as plt

#交差検証とROC曲線、valデータ付き (分割方法はconfig_.yamlで指定）、パラメーター調整なし
with open("./config.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
# group_label = args["group_label"]
# ignore_features = args["ignore_features"]
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])

X = pd.read_csv(args["input_data"])

with open("./pretrained_model.pickle", 'rb') as web:
            model = pickle.load(web)
y_pred = model.predict(X)
y_prob = model.predict_proba(X)
#溶質濃度



#クラスの番号→範囲
def def_class(y_series_1, percentiles = [0, 20, 40, 60, 80]):
    bins = []
    for i in percentiles:
        bins.append(np.percentile(y_series_1.unique(), i))
    bins.append(np.inf)
    print(bins)
    return bins
def cut_to_bins(y_series_1, bins, target_col):
    df_bins=pd.DataFrame(y_series_1)
    bins_names = range(len(bins)-1)
    df_bins=pd.cut(df_bins.iloc[:,0], bins ,labels=bins_names,right=False)
    df_bins=df_bins.to_frame()
    df_bins=df_bins.rename(columns={df_bins.columns[0]: f"{target_col}_class"})
    return df_bins[f"{target_col}_class"]

def value_to_class(ex_series, target_col, percentile, df_input):
    bins = def_class(df_liquid[target_col], percentile, )
    series_bins = cut_to_bins(ex_series, bins, target_col)
    df_input[f"{target_col}_class"] = series_bins
    return df_input, bins
pH_bins = [0, 7, 8]
temp_bins = [0, 25, 38,]
bins_class = [range(5), range(5), range(5), pH_bins, temp_bins]
files =f"../data/{file_name}.csv"
df = pd.read_csv(files, index_col=False) 
df = df[df[target_label].isin(mor_class)].drop_duplicates().reset_index(drop=True) #予測する形態指定(solute, liquid, solid)
df_liquid = df[df.mor_label == 1]
target_col = ["rna_conc_log", 'protein_conc_log','ionic_strength',  'pH', 'temp',]
bins_dict = {}
for n, col in enumerate(target_col):
    df, bins = value_to_class(df[col].values, col, bins_class[n], df)
    bins_dict[col] = bins

df_pred = pd.DataFrame(y_pred)
df_prob = pd.DataFrame(y_prob)
pH_decoder = {0: f"1-{pH_bins[1]}", 1: f"{pH_bins[1]}-{pH_bins[2]}", 2: f"{pH_bins[2]}-14"}
temp_decoder = {0: f"{temp_bins[0]}-{temp_bins[1]}", 1: f"{temp_bins[1]}-{temp_bins[2]}", 2: f"{temp_bins[2]}-"}
def bins_decoder(target = "ionic_strength"):
    is_bins = bins_dict[target]
    is_decoder = {0: f"{is_bins[0]}-{is_bins[1]}", 1: f"{is_bins[1]}-{is_bins[2]}", 2: f"{is_bins[2]}-{is_bins[3]}", 3: f"{is_bins[3]}-{is_bins[4]}", 4: f"{is_bins[4]}-"}
    return is_decoder
is_decoder = bins_decoder()
protein_conc_log_decoder = bins_decoder("protein_conc_log")
rna_conc_log_decoder= bins_decoder("rna_conc_log")
# %%

# %%
for i in range(df_pred.shape[0]):
    print("##############")
    print("pH: ", pH_decoder[int(df_pred.loc[i, 0])])
    print("Temperature: ",temp_decoder[int(df_pred.loc[i, 1])])
    print("Ionic strength: ",is_decoder[int(df_pred.loc[i, 2])])
    print("Protein conc. (log[uM]): ", protein_conc_log_decoder[int(df_pred.loc[i, 3 ])])
    print("RNA conc. (log[uM]): ", rna_conc_log_decoder[int(df_pred.loc[i, 4])])

    df_pred.loc[i, "pH_decode"] = pH_decoder[int(df_pred.loc[i, 0])]
    df_pred.loc[i, "temp_decode"] = temp_decoder[int(df_pred.loc[i, 1])]
    df_pred.loc[i, "is_decode"] = is_decoder[int(df_pred.loc[i, 2])]
    df_pred.loc[i, "protein_conc_log_decoder"] = protein_conc_log_decoder[int(df_pred.loc[i, 3 ])]
    df_pred.loc[i, "rna_conc_log_decoder"] = rna_conc_log_decoder[int(df_pred.loc[i, 4])]
# %%
df_pred.to_csv("./result.csv", index=False)
# %%
