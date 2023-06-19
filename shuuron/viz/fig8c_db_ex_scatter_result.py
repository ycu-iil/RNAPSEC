import pandas as pd
import matplotlib.pyplot as plt
import os
"protein conc. vs rna conc., 実験結果ごとに色分け"
df_chin = pd.read_csv("../data/chin_all_col.csv", index_col=False)
mor_class = [0, 1]
mor = [["solute", "liquid", "solid"][i] for i in mor_class]
df_chin = df_chin[df_chin["mor_label"].isin(mor_class)] 
os.makedirs("./plt_database/", exist_ok=True)

##実験結果により色分け: chinのみ
x_ex = "protein_conc_log" 
y_ex = "rna_conc_log"
title = "Solute concentration \n colored by results"
x_label = "Log [Protein] (μM)"
y_label = "Log [RNA] (μM)"

chin_liquid = df_chin[df_chin.mor_label == 1]
chin_solute = df_chin[df_chin.mor_label == 0]

fig = plt.figure(figsize = (8, 8))
plt.grid(True)    
plt.scatter(x = chin_liquid[f"{x_ex}"], y = chin_liquid[f"{y_ex}"], label = "Liquid", s=30, alpha = 0.4)
plt.scatter(x = chin_solute[f"{x_ex}"], y = chin_solute[f"{y_ex}"], label = "Solute", s=30, color = "darkorange", alpha = 0.5)

plt.title(f"{title}", fontsize = 25, pad = 10)
plt.legend(fontsize = 20, loc =
           "lower left")
plt.ylabel(ylabel = y_label, fontsize =20)
plt.xlabel(xlabel = x_label, fontsize =20)
plt.show()
fig.savefig(f"./plt_database/fig8c_scatter_{x_ex}_{y_ex}_colored_result.png")