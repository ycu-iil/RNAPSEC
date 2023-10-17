from sklearn import preprocessing
import numpy as np
import pandas as pd
def x_use_col(df):
    X = df.loc[:, ['protein_weight', 'protein_A', 'protein_C', 'protein_D', 'protein_E',
       'protein_F', 'protein_G', 'protein_H', 'protein_I', 'protein_K',
       'protein_L', 'protein_M', 'protein_N', 'protein_P', 'protein_Q',
       'protein_R', 'protein_S', 'protein_T', 'protein_V', 'protein_W',
       'protein_Y', 'protein_aromaticity', 'protein_instability',
       'protein_gravy', 'protein_average', 'protein_isopt', 'protein_helix',
       'protein_turn', 'protein_sheet',
        'rna_rna_A', 'rna_rna_C', 'rna_rna_G', 'rna_rna_U', 'rna_rna_AA', 'rna_rna_AC',
        'rna_rna_AG', 'rna_rna_AU', 'rna_rna_CA', 'rna_rna_CC', 'rna_rna_CG', 'rna_rna_CU', 'rna_rna_GA',
        'rna_rna_GC', 'rna_rna_GG', 'rna_rna_GU', 'rna_rna_UA', 'rna_rna_UC', 'rna_rna_UG', 'rna_rna_UU',
        'rna_rna_AAA', 'rna_rna_AAC', 'rna_rna_AAG', 'rna_rna_AAU', 'rna_rna_ACA', 'rna_rna_ACC',
        'rna_rna_ACG', 'rna_rna_ACU', 'rna_rna_AGA', 'rna_rna_AGC', 'rna_rna_AGG', 'rna_rna_AGU',
        'rna_rna_AUA', 'rna_rna_AUC', 'rna_rna_AUG', 'rna_rna_AUU', 'rna_rna_CAA', 'rna_rna_CAC',
        'rna_rna_CAG', 'rna_rna_CAU', 'rna_rna_CCA', 'rna_rna_CCC', 'rna_rna_CCG', 'rna_rna_CCU',
        'rna_rna_CGA', 'rna_rna_CGC', 'rna_rna_CGG', 'rna_rna_CGU', 'rna_rna_CUA', 'rna_rna_CUC',
        'rna_rna_CUG', 'rna_rna_CUU', 'rna_rna_GAA', 'rna_rna_GAC', 'rna_rna_GAG', 'rna_rna_GAU',
        'rna_rna_GCA', 'rna_rna_GCC', 'rna_rna_GCG', 'rna_rna_GCU', 'rna_rna_GGA', 'rna_rna_GGC',
        'rna_rna_GGG', 'rna_rna_GGU', 'rna_rna_GUA', 'rna_rna_GUC', 'rna_rna_GUG', 'rna_rna_GUU',
        'rna_rna_UAA', 'rna_rna_UAC', 'rna_rna_UAG', 'rna_rna_UAU', 'rna_rna_UCA', 'rna_rna_UCC',
        'rna_rna_UCG', 'rna_rna_UCU', 'rna_rna_UGA', 'rna_rna_UGC', 'rna_rna_UGG', 'rna_rna_UGU',
        'rna_rna_UUA', 'rna_rna_UUC', 'rna_rna_UUG', 'rna_rna_UUU',
        'rna_fickett_score-ORF', 'rna_fickett_score-full-sequence',
       'rna_tsallis_k1', 'rna_tsallis_k2', 'rna_real_avg', 'rna_real_peak',
       'rna_binary_avg', 'rna_binary_peak', 'rna_zcurve_avg',
       'rna_zcurve_peak', 'rna_shanon_k1', 'rna_shanon_k2',
       'rna_cv_ORF_length']]
    return X
def read_file():
    return pd.read_excel("../example.xlsx", engine = "openpyxl", index_col=False)
df_input = read_file()
df_rna = pd.read_csv("./prepro_results/preprocessing_result_rna.csv").reset_index(drop = True)
df_b = pd.read_csv("./prepro_results/preprocessing_result_aa.csv").reset_index(drop = True)
assert df_rna.shape[0] == df_b.shape[0]== df_input.shape[0], "shape of input files did not matched"
df = pd.concat([ df_b, df_rna], axis = 1)
X = x_use_col(df)
X.to_csv("../input.csv", index=False)