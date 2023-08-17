#mathfeatureの実行結果農地モデルに使うやつのみをcsvファイルに吐き出させるところまで
import pandas as pd

#fickett score -> 特にnanや不要なカラムは含まれていない
df_fickett_score0 = pd.read_csv("./mf_result/mf_fickett.csv")
df_fickett2 = df_fickett_score0.loc[:, ['fickett_score-ORF', 'fickett_score-full-sequence']]
df_fickett2 = df_fickett2[pd.to_numeric(df_fickett2["fickett_score-ORF"], errors="coerce").notna()]
#k-mer -> 特にnanや不要なカラムは含まれていない
df_3mer_0 = pd.read_csv("./mf_result/mf_3mer.csv" )
df_kmer = df_3mer_0[df_3mer_0.AA.notna()]
df_kmer = df_kmer[pd.to_numeric(df_kmer.A, errors="coerce").notna()].drop(["nameseq", "label"], axis = "columns")
assert df_kmer[~pd.to_numeric(df_kmer.A, errors="coerce").notna()].shape[0] == 0, "kmer included non int value"

df_orf = pd.read_csv("./mf_result/mf_orf.csv")
df_orf_used=df_orf.cv_ORF_length
df_orf_used = df_orf_used[pd.to_numeric(df_orf_used, errors="coerce").notna()]
assert df_orf_used[~pd.to_numeric(df_orf_used, errors="coerce").notna()].shape[0] == 0, "ORF included non int value"
#entropy
df_tsallis = pd.read_csv("./mf_result/mf_tsallis_entropy.csv")
df_tsallis2_1=df_tsallis.reset_index(drop=True)
df_tsallis3 = df_tsallis2_1.rename(columns = {"k1" : "tsallis_k1", "k2" : "tsallis_k2"})
df_tsallis3 = df_tsallis3.loc[:, ["tsallis_k1", "tsallis_k2"]]
df_tsallis3 = df_tsallis3[pd.to_numeric(df_tsallis3.tsallis_k1, errors="coerce").notna()]
assert df_tsallis3[~pd.to_numeric(df_tsallis3.tsallis_k1, errors="coerce").notna()].shape[0] == 0, "kmer included non int value"

df_shanon0 = pd.read_csv("./mf_result/mf_shannon.csv")
df_shanon = df_shanon0.drop(["nameseq", "label"], axis="columns")
df_shanon = df_shanon.rename(columns={"k1": "shanon_k1", "k2" : "shanon_k2"})
df_shanon = df_shanon[pd.to_numeric(df_shanon.shanon_k1, errors="coerce").notna()]
assert df_shanon[~pd.to_numeric(df_shanon.shanon_k1, errors="coerce").notna()].shape[0] == 0, "kmer included non int value"

#fourier
df_fourier_binary = pd.read_csv("./mf_result/mf_fourier_binary.csv")
df_fourier_real = pd.read_csv("./mf_result/mf_fourier_real.csv")
df_fourier_zcurve = pd.read_csv("./mf_result/mf_fourier_zcurve.csv")

df_fourier_zcurve=df_fourier_zcurve.rename(columns={"average": "zcurve_avg"})
df_fourier_zcurve=df_fourier_zcurve.rename(columns={"peak": "zcurve_peak"})
df_fourier_real=df_fourier_real.rename(columns={"average": "real_avg"})
df_fourier_real=df_fourier_real.rename(columns={"peak": "real_peak"})
df_fourier_binary=df_fourier_binary.rename(columns={"average": "binary_avg"})
df_fourier_binary=df_fourier_binary.rename(columns={"peak": "binary_peak"})
df_fourier=pd.concat([df_fourier_real.loc[:, ["real_avg", "real_peak"]], 
                     df_fourier_binary.loc[:, ["binary_avg", "binary_peak"]], 
                     df_fourier_zcurve.loc[:, ["zcurve_avg", "zcurve_peak"]]], axis="columns")
df_fourier = df_fourier[pd.to_numeric(df_fourier.zcurve_avg, errors="coerce").notna()]
assert df_fourier[~pd.to_numeric(df_fourier.zcurve_avg, errors="coerce").notna()].shape[0] == 0, "kmer included non int value"
#filterなし、RNAのkmer全部
for i in df_kmer.columns:
    df_kmer = df_kmer.rename(columns = { f"{i}" : f"rna_{i}"})

#concat, rename
df_rna_features=pd.concat([df_kmer, df_fickett2, df_tsallis3, df_fourier, df_shanon, df_orf_used, ], axis="columns")
df_rna_features_2 = df_rna_features.copy()
assert df_rna_features_2.columns.str.contains("Unnamed", na = True).sum()== 0, "Unnamed columns included in the output file"

for i in df_rna_features.columns:
    df_rna_features_2 = df_rna_features_2.rename(columns = { f"{i}" : f"rna_{i}"})
    assert df_rna_features_2[df_rna_features_2[ f"rna_{i}"].isna()].shape[0]==0, "nan included in rna features"
df = pd.read_csv("./preprocessing_results/rnap_preprocessing_1.csv",index_col=False)
assert df_rna_features_2.shape[0]  == df.shape[0], "index of the output file dose'nt match to the output file of rnap_prepro_ex_1.py. "
df_rna_features_2.to_csv("./preprocessing_results/rna_mf_results.csv", index=False)

