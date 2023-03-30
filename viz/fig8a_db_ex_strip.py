import argparse
import numpy as np
import pandas as pd
import yaml
import glob
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut, cross_val_score, cross_validate
import pickle
import os
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

with open("../main_exdata_1212/config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
fold_strategy = args["fold_strategy"]
val_fold_strategy = args["val_fold_strategy"]
n_splits = args["n_splits"]
random_state = args["random_state"]
data_dir = args["data_dir"]
file_name = args["file_name"]
mor_class = np.array(args["mor_class"])
data_dict = {}
X_dict = {}
y_dict = {}
group_dict = {}

max_num = args["max_num"]
min_num = args["min_num"]
early_stopping_num = args["early_stopping"]
val_fold = args["val_fold"]
files_path = data_dir + file_name 


df_chin = pd.read_csv("../data/chin_all_col.csv", index_col=False)
df_prepro = pd.read_csv("../data/prepro_all_col.csv", index_col=False)
df_chin = df_chin.rename(columns = {"index": "ini_idx"})
df_default = pd.read_csv("/home/chin/rnaphasep_vs/clf_mor/model/model_0222/dataset_0222/original/default_dataset_0221.csv")
mor_class = [0, 1]
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_chin = df_chin[df_chin["mor_label"].isin(mor_class)] 
df_prepro = df_prepro[df_prepro["mor_label"].isin(mor_class)] 

#plotly で実験条件の分布をstrip plotで描画
RENDERER = 'plotly_mimetype+notebook'
def show_fig(fig):
    """Jupyter Bookでも表示可能なようRendererを指定"""
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.show(renderer=RENDERER)

df_prepro["axis"] = "RNAPhaSep"
df_chin["axis"] = "RNAPSEC"
df = pd.concat([df_chin.loc[:, ["protein_conc_log", "axis"]],df_prepro.loc[:, ["protein_conc_log", "axis"]]], axis = "index")

def plotly_strip(x_col = "protein_conc_log", title = "Protein concentration", x_axis = "Protein conc. [log (μM)]"):
    df = pd.concat([df_prepro.loc[:, [f"{x_col}", "axis"]], df_chin.loc[:, [f"{x_col}", "axis"]]], axis = "index")
    fig = px.strip(
        df, x='axis', y=f'{x_col}',
        title=f'{title}', height=600, width = 400, color = "axis")
    fig.update_traces(marker={'line_width':1, 'opacity':0.7})
    fig.update_yaxes(linecolor='black', title=f'{x_axis}', gridcolor='gray', mirror=True)
    fig.update_xaxes(linecolor='black', title='DataSet', gridcolor='gray', mirror=True)
    fig.update_layout(plot_bgcolor = "white", title = dict(x=0.55,y = 0.95, xanchor='center', 
                                                           font = dict(size = 20),), 
                      showlegend = False,
                     xaxis = dict(title = dict(font = dict(size = 20)), tickfont = dict(size = 20)), 
                      yaxis = dict(title = dict(font = dict(size = 20)),tickfont = dict(size = 20)))
    show_fig(fig)
    fig.write_image(f"./plt_database/ex_strip_{title}.png")
    return

exs = ["protein_conc_log", "rna_conc_log", "ionic_strength", "pH", "temp"]
fig_titles = ["Protein concentration", "RNA concentration", "Ionic strength", "pH", "Temperature"]
x_label = ["Log [Protein] (μM)", "Log [RNA] (μM)", "Ionic strength", "pH", "Temperature [℃]"]
for ex, title, x_label in zip(exs, fig_titles, x_label):
#     plt_vio_ex(ex_col=ex, title=title)
    plotly_strip(x_col = ex, title = title, x_axis = x_label)
print("end")