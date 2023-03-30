import pandas as pd
import matplotlib.pyplot as plt
import os
"protein conc. vs 他の実験条件。データセット間の比較"
df_chin = pd.read_csv("../data/chin_all_col.csv", index_col=False)
df_prepro = pd.read_csv("../data/prepro_all_col.csv", index_col=False)
mor_class = [0, 1]
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_chin = df_chin[df_chin["mor_label"].isin(mor_class)] 
df_prepro = df_prepro[df_prepro["mor_label"].isin(mor_class)] 
df_chin_only = df_chin[~((df_chin.protein_conc_log.isin(df_prepro.protein_conc_log)) &
                         (df_chin.rna_conc_log.isin(df_prepro.rna_conc_log)))]
df_chin_dup = df_chin[((df_chin.protein_conc_log.isin(df_prepro.protein_conc_log)) &
                         (df_chin.rna_conc_log.isin(df_prepro.rna_conc_log)))]
df_prepro_only = df_prepro[~((df_prepro.protein_conc_log.isin(df_chin.protein_conc_log)) & 
                             (df_prepro.rna_conc_log.isin(df_chin.rna_conc_log)))]
df_prepro_dup =  df_prepro[((df_prepro.protein_conc_log.isin(df_chin.protein_conc_log)) & 
                             (df_prepro.rna_conc_log.isin(df_chin.rna_conc_log)))]
os.makedirs("./plt_database/", exist_ok=True)

x_ex = "protein_conc_log" 
y_ex = "rna_conc_log"
title = "solute concentration"
x_label = "Protein conc.[log(μM)]"
y_label = "RNA conc. [log(μM)]"
#どのデータセットに含まれてるかによって色分け
def colored_dup(x_ex = "protein_conc_log", y_ex = "rna_conc_log", title = "solute concentration", x_label = "Protein conc.[log(μM)]", y_label = "RNA conc. [log(μM)]"):
    df_chin_only = df_chin[~((df_chin[x_ex].isin(df_prepro[x_ex])) &
                             (df_chin[y_ex].isin(df_prepro[y_ex])))]
    df_chin_dup = df_chin[((df_chin[x_ex].isin(df_prepro[x_ex])) &
                             (df_chin[y_ex].isin(df_prepro[y_ex])))]
    df_prepro_only = df_prepro[~((df_prepro[x_ex].isin(df_chin[x_ex])) & 
                                 (df_prepro[y_ex].isin(df_chin[y_ex])))]
    df_prepro_dup =  df_prepro[((df_prepro[x_ex].isin(df_chin[x_ex])) & 
                                 (df_prepro[y_ex].isin(df_chin[y_ex])))]


    fig = plt.figure(figsize = (10, 10))
    plt.grid(True)    
    plt.scatter(x = df_chin_only[f"{x_ex}"], y = df_chin_only[f"{y_ex}"], label = "RNAPSEC only", s=30, alpha = 0.5, color = "dodgerblue")
    plt.scatter(x = df_prepro_only[f"{x_ex}"], y = df_prepro_only[f"{y_ex}"], label = "RNAPhaSep only", s=30, color = "orange", alpha = 0.5)
    plt.scatter(x = df_prepro_dup[f"{x_ex}"], y = df_prepro_dup[f"{y_ex}"], label = "Both", s=30, alpha = 0.5, color = "black", marker = "D")
    plt.title(f"{title}", fontsize = 32, pad = 20)
    plt.legend(fontsize = 28,  bbox_to_anchor=(1, 1))
    plt.ylabel(ylabel = y_label, fontsize =28, labelpad = 10)
    plt.xlabel(xlabel = x_label, fontsize =28, labelpad = 10)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    plt.show()
    fig.savefig(f"./plt_database/fig8b_scatter_{x_ex}_{y_ex}.png")
    return
exs = ["protein_conc_log", "rna_conc_log", "ionic_strength", "pH", "temp"]
fig_titles = ["Protein concentration", "RNA concentration", "Ionic strength", "pH", "Temperature"]
x_labels = ["Log [Protein] (μM)", "Log [RNA] (μM)", "Ionic strength", "pH", "Temperature [℃]"]
for ex, title, x_label in zip(exs, fig_titles, x_labels):
    for y_ex, title, y_label in zip(exs, fig_titles, x_labels):
        colored_dup(x_ex = ex, y_ex = y_ex, x_label= x_label, y_label = y_label,title = f"{ex} & {y_ex}")
print("end")