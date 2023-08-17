#%%
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import molecular_weight
import numpy as np
import os
# %%
def read_file():
    return pd.read_excel("./example.xlsx", engine = "openpyxl", index_col=False)
df = read_file()
os.makedirs("./prepro_results/", exist_ok=True)
def prepro_conc(conc_value = "protein_conc"): #pH, temp, ionic_strength 済み
    df[conc_value] = df[conc_value].astype("float")
    return df
value_cols = ["protein_conc", "rna_conc", "pH", "temperature", "ionic_strength"]
def calc_solute_conc(df):
    unit_list = ["nM", "uM", "mM"] #蛋白質・RNA濃度はモーラーで書いてもらう
    assert df.protein_unit.isin(unit_list).shape[0] == df.shape[0], "protein unit must be [nM, uM, mM]"
    assert df.rna_unit.isin(unit_list).shape[0] == df.shape[0], "RNA unit must be [nM, uM, mM]"
    def convert_to_uM(target = "protein"):
        df[f"{target}_conc"][df[f"{target}_conc"]=="nM"] = df[f"{target}_conc"]/1000
        df[f"{target}_conc"][df[f"{target}_conc"]=="mM"] = df[f"{target}_conc"]*1000
        df[f"{target}_conc_log"] = np.log10(df[f"{target}_conc"].values)
        return df
    df = convert_to_uM("protein")
    df = convert_to_uM("rna")
    return df
df= calc_solute_conc(df)
assert df["protein_conc_log"].notna().shape[0] == df.shape[0], "error: protein concentration calcuration "
assert df["rna_conc_log"].notna().shape[0] == df.shape[0], "error: rna concentration calcuration "

#%%
def convert_protein_sequence_to_features():
    from rnap_prepro_bf import seq2features
    df["aa"] = df.protein_sequence.str.replace("\n", "")
    df_aa_features = seq2features(df.aa.values)
    for col in df_aa_features.columns:
        df_aa_features = df_aa_features.rename(columns={col: f"protein_{col}"})
    return df, df_aa_features
df, df_aa_features = convert_protein_sequence_to_features()
df_aa_features.to_csv("./prepro_results/preprocessing_result_aa.csv", index=False)
df.to_csv("./prepro_results/preprocessing_result_ex.csv", index=False)
def output_rna_fasta():
    def seq_to_record(rna_seq, x, name):    
        seq_r2=SeqRecord(Seq(rna_seq))
        #idが被らないようもとdfのインデックスタンパク質名の先頭に
        a = str(x) # + str(i)
        seq_r2.id=f"{a}"
        seq_r2.name = name
        return seq_r2
    records=[]
    for (seq, index, name) in zip(df.rna_sequence.values,range(df.shape[0]), df.index):
        a=seq_to_record(str(seq), index, name)
        records.append(a)
    file_path = "./prepro_results/rna_seq.fasta"
    SeqIO.write(records,file_path , "fasta")  
    os.makedirs("./mf_result", exist_ok= True)
    return
output_rna_fasta()

# %%
