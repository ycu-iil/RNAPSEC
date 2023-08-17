#%%
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

#溶質濃度
def cut_solute_conc_bins(df):

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


df_liquid = df[df.mor_label == 1]
#クラス分け
df_liquid = cut_solute_conc_bins(df_liquid)
rnapsec_ph = df_liquid.pH
rnapsec_ph_bins, bins = cut_ph_class(rnapsec_ph)
df_liquid["ph_bins"] = rnapsec_ph_bins
rnapsec_temp = df_liquid.temp
rnapsec_temp_bins, bins = cut_bins_class(rnapsec_temp)
df_liquid["temp_bins"] = rnapsec_temp_bins
df_liquid = cut_is_bins(df_liquid, "ionic_strength")
#特徴量と説明変数に分ける
y_rnapsec_liquid = df_liquid.loc[:, ["ph_bins", "temp_bins", "ionic_strength_bins", "protein_conc_log_class", "rna_conc_log_class"]]
X_rnapsec_liquid = df_liquid.drop(["ph_bins", "temp_bins", "protein_conc_log_class", "rna_conc_log_class","ionic_strength_bins", "ionic_strength", 'rna_conc_log', 'protein_conc_log', 'pH', 'temp', 'mor_label', group_label, "rnapsec_all_col_idx", 'protein_sequence', 'rna_sequence', 'protein_name', 'aa_label','aa_rna_label', ], axis = "columns")
clf=lgb.LGBMClassifier(
    feature_pre_filter= False,lambda_l2= 9.5,num_leaves= 9,feature_fraction= 0.984,bagging_fraction= 1.0,bagging_freq= 0,min_child_samples= 5
)
lgb_c = ClassifierChain(clf, order=[0, 1, 2, 3, 4])
lgb_c.fit(X_rnapsec_liquid, y_rnapsec_liquid)
with open (f"../pretrained_model.pickle", mode = "wb") as f:
    pickle.dump(lgb_c, f)
assert X_rnapsec_liquid.shape[1]==126, "X shape dose'nt matched to 126"