
import numpy as np
import pandas as pd
import yaml
import pickle
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def main():

    def create_test_ex(df):
        protein_conc = np.linspace(df[target_col_x].min()-0.5, df[target_col_x].max()+0.5, 20)
        print(protein_conc)
        rna_conc = np.linspace(df[target_col_y].min()-0.5, df[target_col_y].max()+0.5, 20)
        rna_conc.shape
        patterns = []
        for protein in protein_conc:
            for rna in rna_conc:
                patterns.append((protein,rna))
        df_add= pd.DataFrame(patterns)
        
        print(df_add)
        df_add = df_add.rename(columns = {0:target_col_x, 1:target_col_y})
        for col in df.drop([target_col_x, target_col_y], axis = "columns").columns:
            df_add[col] = df[col].iloc[0]
        return df_add.loc[:, df_test_0.drop(["aa_label", "aa_rna_label"], axis = "columns").columns]


    def plt_phase_diagram(df_test, n):
        pred_protein_1 = df_test[(df_test.preds == 1)][target_col_x]
        pred_rna_1 = df_test[(df_test.preds == 1)][target_col_y]
        pred_protein_0 = df_test[(df_test.preds == 0)][target_col_x]
        pred_rna_0 = df_test[(df_test.preds == 0) ][target_col_y]
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.scatter(x = pred_protein_1, y = pred_rna_1, c = "lightskyblue",alpha = 1, label = "Prediction: Liquid", s = 620, marker = "s") # c = "dodgerblue"
        ax.scatter(x = pred_protein_0, y = pred_rna_0, c = "navajowhite", alpha = 1, label = "Prediction: Solute", s =620, marker = "s")
        ax.set_xlabel("Protein conc. (log μM)", fontsize = 26, labelpad = 15)
        ax.set_ylabel("RNA conc. (log μM)", fontsize = 26, labelpad = 15)
        ax.xaxis.set_major_locator(ticker.LinearLocator(6))
        ax.xaxis.set_tick_params(direction='out', labelsize=20, width=3, pad=10)
        ax.yaxis.set_tick_params(direction='out', labelsize = 20, width=3, pad=10)
        ax.legend(fontsize = 28,  loc='lower right', bbox_to_anchor=(1.8, 0))
        ax.set_title(f"Phase diagram", fontsize = 26, pad = 24)
        fig.savefig(f"phase_diagram_{n}.png", bbox_inches='tight', transparent=False)
        plt.close()
        return
        
    def read_data(file_path):
        df = pd.read_csv(file_path, index_col=False)
        return df


    with open("./config.yaml",'r')as f:
        args = yaml.safe_load(f)
    target_col_x = args["target_x"]
    target_col_y = args["target_y"]
    df_target = read_data(args["input_data"])
    with open("./pretrained_model.pickle", 'rb') as web:
            model = pickle.load(web)
    for aa_rna in df_target.aa_rna_label.unique():
        df_test_0 = df_target[df_target.aa_rna_label == aa_rna]
        df_test = create_test_ex(df_test_0.drop(["aa_label", "aa_rna_label"], axis = "columns"))
        X = df_test.values
        
        df_test["preds"] = model.predict(X)
        df_test["proba"] = model.predict_proba(X)[:, 1]
        plt_phase_diagram(df_test=df_test, n = aa_rna)

if __name__== "__main__":
    main()
