#%%
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import molecular_weight
import numpy as np
import os

def read_file():
    return pd.read_excel("./data/rnapsec.xlsx", engine = "openpyxl", index_col=False)
df_rnapsec = read_file()
df_rnapsec = df_rnapsec.drop(df_rnapsec.columns[df_rnapsec.columns.str.contains("Unnamed")], axis = "columns")
df_rnapsec = df_rnapsec[df_rnapsec["rnaphasep_index"].notna()]
df = df_rnapsec[(df_rnapsec['added']=="added")| (df_rnapsec['added']=="edited" )]
df = df[(df.rnaphasep_other_requirement == "-") &( df["component type"] =="RNA + protein")] 
df["aa"] = np.nan
df["aa"][~df.protein_sequence.str.contains("\|")] = df.protein_sequence
df["aa"][df.protein_sequence.str.contains("\|")] = df.protein_sequence.str.split("\n", n = 1, expand = True)[1].str.replace("\n|\||;", "")
df["aa"] = df["aa"].str.replace("\n", "")
df["aa"] = df["aa"].str.replace("|", "")
df["aa"] = df["aa"].str.replace(";", "")
df["aa"] = df["aa"].str.replace("(^[A-Z])","",  regex = False)
# df["aa"] = df["aa"].str.replace(">spQ99496RING2_HUMAN E3 ubiquitin-protein ligase RING2 OS=Homo sapiens OX=9606 GN=RNF2 PE=1 SV=1", "")
df["aa"] = df["aa"].str.replace(">spQ99496RING2_HUMAN E3 ubiquitin-protein ligase RING2 OS=Homo sapiens OX=9606 GN=RNF2 PE=1 SV=1", "") #fusion protein
df = df[df.aa.str.contains("[A-Z][A-Z]", na = False)]
df = df[df.aa.notna()]
print( df[df.aa.str.contains("[0-9]", na = True)].aa.unique())
assert df[df.aa.str.contains("[0-9]", na = True)].shape[0] == 0, "protein seq contains non single letter"
#%%
df
#%%
df1 = df[df.rna_sequence != "-"  ]
df1 = df1[df.rna_sequence.notna()]
df1[df1.rna_sequence.str.contains(";\|-", na = False)] 
df1.rna_sequence = df1.rna_sequence.str.replace("-", "")
df1.rna_sequence = df1.rna_sequence.str.replace(";", "")
df1.rna_sequence = df1.rna_sequence.str.replace("|", "")
assert df1[~df1.rna_sequence.str.contains("[A-Z]", na = True)].shape[0] == 0, "rna seq contains non-single letter"

#濃度
#最初に全部floatに変換する
df1.protein_conc = df1.protein_conc.astype("float")
df1.rna_conc = df1.rna_conc.astype("float")
#濃度欠損のデータをはずす
df1 = df1[(df1.rna_conc.notna()) & (df1.protein_conc.notna())] #
df1 = df1[(df1.rna_conc != 0) & (df1.protein_conc != 0)]
df1["protein_weight"]="?"
for i in df1.index:
    df1.loc.__setitem__((i,"protein_weight"),  ("%0.2f" % molecular_weight(f"{df1.aa.loc[i]}", "protein")))
#単位の表記統一 (μ, 文字化け(元ファイルはμ) → u)
df1.rna_unit =df1.rna_unit.str.replace('(ﾎｼ|u|µ|\_)', "u")
df1.rna_unit = df1.rna_unit.str.replace('_', "u")
df1.protein_unit =df1.protein_unit.str.replace('(ﾎｼ|u|µ|\_)', "u")
df1.protein_unit = df1.protein_unit.str.replace('_', "u")
df1.protein_unit =df1.protein_unit.str.replace('um', "uM")
#タンパク質濃度の単位をuMに統一
protein_conc = df1.loc[:, ['protein_conc','protein_unit',"protein_weight"]]
protein_conc.protein_unit = protein_conc.protein_unit.str.replace('(ﾎｼ|u|µ|\_)', "u")
protein_conc.protein_unit = protein_conc.protein_unit.str.replace('_', "u")
protein_conc["uM"]=np.nan
protein_conc.protein_conc=protein_conc.protein_conc.astype("float")
#計算
protein_conc.uM[protein_conc.protein_unit=="uM"]=protein_conc.protein_conc
protein_conc.uM[protein_conc.protein_unit=="nM"]=protein_conc.protein_conc/1000
protein_conc.uM[protein_conc.protein_unit=="mM"]=protein_conc.protein_conc*1000
protein_conc.protein_weight=protein_conc.protein_weight.astype("float")
protein_conc.uM[(protein_conc.protein_unit=="ng/ul")] = ((protein_conc.protein_conc/1000)/protein_conc.protein_weight)*1000
protein_conc.uM[(protein_conc.protein_unit=="mg/ml")|(protein_conc.protein_unit=="ug/uL")|(protein_conc.protein_unit=="ug/ul")|(protein_conc.protein_unit=="ng/nl")] = ((protein_conc.protein_conc)/protein_conc.protein_weight)*1000

##RNA濃度
df1.rna_unit =df1.rna_unit.str.replace('(ﾎｼ|u|µ|\_)', "u") #消すと全落ちする
df1.rna_unit =df1.rna_unit.str.replace('_', "u") 
df1_1=df1[df1.rna_sequence.notna()]
df1_1["rna_weight"]="?"

for i in df1_1.index:
    df1_1.loc.__setitem__((i,"rna_weight"),  ("%0.2f" % molecular_weight(
        f"{df1_1.rna_sequence.loc[i]}", "RNA")))
rna_conc=df1_1.loc[:, ['rna_conc', 'rna_unit', 'rna_weight']]

rna_conc["rna_uM"]=np.nan
rna_conc.rna_conc = rna_conc.rna_conc.astype("float")
rna_conc.rna_uM[rna_conc.rna_unit=="uM"]=rna_conc.rna_conc
rna_conc.rna_uM[rna_conc.rna_unit=="nM"]=rna_conc.rna_conc/1000
rna_conc.rna_uM[rna_conc.rna_unit=="mM"]=rna_conc.rna_conc*1000
#g/l勢-> uM: [g/l]/[molecular_weight] * 1000
rna_conc.rna_weight=rna_conc.rna_weight.astype("float")

rna_conc.rna_uM[(rna_conc.rna_unit=="ng/ul")|(rna_conc.rna_unit=="ng/uL")] = ((rna_conc.rna_conc/1000)/rna_conc.rna_weight)*1000
rna_conc.rna_uM[(rna_conc.rna_unit=="mg/ml")|(rna_conc.rna_unit=="ug/uL")|(rna_conc.rna_unit=="ug/ul")|(rna_conc.rna_unit=="ng/nl")] = ((rna_conc.rna_conc)/rna_conc.rna_weight)*1000
rna_conc.rna_uM[(rna_conc.rna_unit=='ug/ml')] = ((rna_conc.rna_conc)/(rna_conc.rna_weight*1000))*1000

conc =pd.concat([protein_conc.uM, rna_conc.rna_uM], axis="columns")
df1_1["rna_conc_uM"] = rna_conc.rna_uM
df1_1["protein_conc_uM"] = conc.uM

#uM -> log10(uM)
df1_1["rna_conc_log"] = np.log10(rna_conc.rna_uM.values)
df1_1["protein_conc_log"] = np.log10(protein_conc.uM.values)
assert df1_1[(df1_1.rna_conc_log == (-np.inf))].shape[0]==0, "rna concentration included nan or 0"
assert df1_1[(df1_1.protein_conc_log == (-np.inf))].shape[0]==0, "protein concentration included nan or 0"

df1_1["temp"] = df1_1.temperature.str.replace("RT", "25").fillna(df1_1.temperature)
df1_1["temp"] = df1_1.temp.str.extract("(\d*)")
df1_1["temp"] =df1_1.temp.fillna(df1_1[df1_1.temp.isna()].temperature.astype("float"))
#temperature->摂氏・室温は25とする。数値だけのカラムを作成
df1_1["temp"] = df1_1.temp.astype("float")
df1_1.temp.unique()

#%%
df1_1.morphology_add.value_counts()

#%%
assert df1_1.temp.max() <= 100, "temp included value over 100"

#morphology_add(追加項目)が書き忘れているやつをmorphology(元の項目)で埋める
assert df1_1[df1_1.morphology_add.str.contains("-", na = True)].shape[0] == 0, "morphology contains nan"
df1_1.morphology_add = df1_1.morphology_add.replace('-', np.nan)
df1_1.morphology_add= df1_1.morphology_add.fillna(df1_1.rnaphasep_morphology)
df1_1 = df1_1[~df1_1.morphology_add.str.contains(",")]

df1_1
#np.infは後でdrop,
def seq_to_record(rna_seq, x, name):    
    seq_r2=SeqRecord(Seq(rna_seq))
    #idが被らないようもとdfのインデックスタンパク質名の先頭に
    a = str(x) # + str(i)
    seq_r2.id=f"{a}"
    seq_r2.name = name
    return seq_r2
records=[]
for (seq, index, name) in zip(df1_1.rna_sequence.values,range(df1_1.shape[0]), df1_1.index):
    a=seq_to_record(str(seq), index, name)
    records.append(a)

os.makedirs("./preprocessing_results/", exist_ok=True)
df1_1.drop("protein_weight", axis = "columns").to_csv("./preprocessing_results/rnap_preprocessing_1.csv", index=True, index_label="rnapsec_idx")
file_path = "./data/rna_seq.fasta"
assert not os.path.isfile(file_path), "file name: rna_seq.fasta was already existed"
SeqIO.write(records,file_path , "fasta")  
os.makedirs("./mf_result", exist_ok= True)
