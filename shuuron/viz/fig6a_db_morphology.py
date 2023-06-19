import argparse
import numpy as np
import pandas as pd
import yaml
import pickle
import os
import matplotlib.pyplot as plt

"各論文の収録データ数を棒グラフで図示"
#RNAPhaSep (Curated)とRNAPSEC (Curated)
with open("../lgb_opt/config_.yaml",'r')as f:
    args = yaml.safe_load(f)

target_label = args["target_label"]
group_label = args["group_label"]
ignore_features = args["ignore_features"]
mor_class = np.array(args["mor_class"])

df_default = pd.read_csv("/home/chin/rnaphasep_vs/clf_mor/model/model_0222/dataset_0222/original/default_dataset_0221.csv")
df_default = df_default.drop(df_default.columns[df_default.columns.str.contains("Unnamed")], axis = "columns")
df_default_component_1 = df_default[df_default.components_type == ("RNA + protein")]

df_chin = pd.read_csv("../data/chin_all_col.csv", index_col=False)
df_prepro = pd.read_csv("../data/prepro_all_col.csv", index_col=False)
#chin
files = "../data/chin_all_col.csv"
df_ml_chin = pd.read_csv(files, index_col=False) 
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_ml_chin = df_ml_chin[df_ml_chin[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
#prepro
files = "../data/prepro_all_col.csv"
ml_prepro = pd.read_csv(files, index_col=False) 
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_ml_prepro_mor = ml_prepro[ml_prepro[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)

df_mor = pd.concat([df_default_component_1.mor2.value_counts(), df_chin.morphology_add.value_counts(),df_ml_prepro_mor.morphology_add.value_counts(), df_ml_chin.morphology_add.value_counts()], axis = "columns")
#plot用のデータセットをconcat
df_mor.columns = ["RNAPhaSep\n(DB)", "RNAPSEC\n(DB)", "RNAPhaSep\n(Curated)","RNAPSEC\n(Curated)"]
df_mor = df_mor.loc[['solute','liquid', 'gel', 'solid', "else", "unknown"],["RNAPSEC\n(Curated)", "RNAPhaSep\n(Curated)","RNAPSEC\n(DB)",  "RNAPhaSep\n(DB)",]]
df_mor.index = ['Solute','Liquid', 'Gel', 'Solid', 'Others', 'Unknown']
df_mor_t = df_mor.T

#plotlyでplt
import plotly.express as px
RENDERER = 'plotly_mimetype+notebook'
def show_fig(fig):
    """Jupyter Bookでも表示可能なようRendererを指定"""
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.show(renderer=RENDERER)
fig = px.bar(
    df_mor_t, x=df_mor_t.columns, y=df_mor_t.index,
    color_discrete_sequence= ["rgb(166,206,227)", "rgb(31,120,180)",  "rgb(178,223,138)",
                              "rgb(51,160,44)", "rgb(251,154,153)",  "rgb(227,26,28)"],
    barmode='stack', 
    title='各データベースにおける実験結果',width=700, height= 400)
fig.update_layout(dict(font=dict(size=15,
                                       color='black')))
fig.update_xaxes(title=dict(text = 'データ数', font = {"size" : 20}))
fig.update_yaxes(title=dict(text = '', font = {"size" : 20}))
show_fig(fig)