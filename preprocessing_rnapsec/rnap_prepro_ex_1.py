

import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis, molecular_weight

import numpy as np
import codecs
import os
"追加分だけに絞る→タンパク質配列だけのカラム作る→RNA配列をfastaに書き出す→濃度の単位をμMに統一→温度・pHをfloat変換→csvに書き出す"
#元データで前処理しなきゃならん項目を書き出しとく (preproのディレクトにreadmeでかいとく)
with codecs.open("./data/rnapsec.csv", "r", "Shift-JIS", "ignore") as file:
    rnaphasep = pd.read_table(file, delimiter=",")
rnaphasep = rnaphasep.drop(rnaphasep.columns[rnaphasep.columns.str.contains("Unnamed")], axis = "columns")
rnaphasep = rnaphasep[rnaphasep["index"].notna()]

#追加したデータだけに絞る
df_rnaphasep = rnaphasep[(rnaphasep['added']=="added")| (rnaphasep['added']=="edited" )]

df_rnaphasep["aa"] = np.nan
df_rnaphasep["aa"][~df_rnaphasep.protein_sequence.str.contains("\|")] = df_rnaphasep.protein_sequence
df_rnaphasep["aa"][df_rnaphasep.protein_sequence.str.contains("\|")] = df_rnaphasep.protein_sequence.str.split("\n", 1, expand = True)[1].str.replace("\n|\||;", "")
df_rnaphasep["aa"] = df_rnaphasep["aa"].str.replace("(^[A-Z])","",  regex = False)
df_rnaphasep["aa"] = df_rnaphasep["aa"].str.replace(">spQ99496RING2_HUMAN E3 ubiquitin-protein ligase RING2 OS=Homo sapiens OX=9606 GN=RNF2 PE=1 SV=1", "") #fusion protein
df_rnaphasep = df_rnaphasep[df_rnaphasep.aa.str.contains("[A-Z][A-Z]", na = False)]
df_rnaphasep = df_rnaphasep[df_rnaphasep.aa.notna()]
df_rnaphasep = df_rnaphasep.reset_index(drop = False)

df_rnaphasep1 = df_rnaphasep[df_rnaphasep.rna_sequence != "-"  ]
df_rnaphasep1 = df_rnaphasep1[df_rnaphasep.rna_sequence.notna()]

#seqなし、";|-"取り除く
df_rnaphasep1[df_rnaphasep1.rna_sequence.str.contains(";\|-", na = False)] #'UAGAAAACAUGAGGAUCACCCAUGUCUGCAG;|-'
df_rnaphasep1.rna_sequence = df_rnaphasep1.rna_sequence.str.replace(";\|-", "")


#濃度
#最初に全部floatに変換する
df_rnaphasep1.protein_conc = df_rnaphasep1.protein_conc.astype("float")
df_rnaphasep1.rna_conc = df_rnaphasep1.rna_conc.astype("float")
#濃度欠損のデータをはずす
    #na, 0 を含むデータをはずす
df_rnaphasep1 = df_rnaphasep1[(df_rnaphasep1.rna_conc.notna()) & (df_rnaphasep1.protein_conc.notna())] #
df_rnaphasep1 = df_rnaphasep1[(df_rnaphasep1.rna_conc != 0) & (df_rnaphasep1.protein_conc != 0)]
df_rnaphasep1["protein_weight"]="?"
for i in df_rnaphasep1.index:
    df_rnaphasep1.loc.__setitem__((i,"protein_weight"),  ("%0.2f" % molecular_weight(f"{df_rnaphasep1.aa.loc[i]}", "protein")))
#単位の表記統一 (μ, 文字化け(元ファイルはμ) → u)
df_rnaphasep1.rna_unit =df_rnaphasep1.rna_unit.str.replace('(ﾎｼ|u|µ|\_)', "u")
df_rnaphasep1.rna_unit = df_rnaphasep1.rna_unit.str.replace('_', "u")
df_rnaphasep1.protein_unit =df_rnaphasep1.protein_unit.str.replace('(ﾎｼ|u|µ|\_)', "u")
df_rnaphasep1.protein_unit = df_rnaphasep1.protein_unit.str.replace('_', "u")
df_rnaphasep1.protein_unit =df_rnaphasep1.protein_unit.str.replace('um', "uM")
#タンパク質濃度の単位をuMに統一
protein_conc = df_rnaphasep1.loc[:, ['protein_conc','protein_unit',"protein_weight"]]
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
protein_conc.protein_conc[(protein_conc.protein_unit=="mg/ml")|(protein_conc.protein_unit=="ug/uL")|(protein_conc.protein_unit=="ug/ul")|(protein_conc.protein_unit=="ng/nl")] = ((protein_conc.protein_conc)/protein_conc.protein_weight)*1000

##RNA濃度
df_rnaphasep1.rna_unit =df_rnaphasep1.rna_unit.str.replace('(ﾎｼ|u|µ|\_)', "u") #消すと全落ちする
df_rnaphasep1.rna_unit =df_rnaphasep1.rna_unit.str.replace('_', "u") 
df_rnaphasep1_1=df_rnaphasep1[df_rnaphasep1.rna_sequence.notna()]
df_rnaphasep1_1["rna_weight"]="?"

for i in df_rnaphasep1_1.index:
    df_rnaphasep1_1.loc.__setitem__((i,"rna_weight"),  ("%0.2f" % molecular_weight(
        f"{df_rnaphasep1_1.rna_sequence.loc[i]}", "RNA")))
rna_conc=df_rnaphasep1_1.loc[:, ['rna_conc', 'rna_unit', 'rna_weight']]

rna_conc["rna_um"]=np.nan
rna_conc.rna_conc = rna_conc.rna_conc.astype("float")
rna_conc.rna_um[rna_conc.rna_unit=="uM"]=rna_conc.rna_conc
rna_conc.rna_um[rna_conc.rna_unit=="nM"]=rna_conc.rna_conc/1000
rna_conc.rna_um[rna_conc.rna_unit=="mM"]=rna_conc.rna_conc*1000
#g/l勢-> uM: [g/l]/[molecular_weight] * 1000
rna_conc.rna_weight=rna_conc.rna_weight.astype("float")

rna_conc.rna_um[(rna_conc.rna_unit=="ng/ul")|(rna_conc.rna_unit=="ng/uL")] = ((rna_conc.rna_conc/1000)/rna_conc.rna_weight)*1000
rna_conc.rna_um[(rna_conc.rna_unit=="mg/ml")|(rna_conc.rna_unit=="ug/uL")|(rna_conc.rna_unit=="ug/ul")|(rna_conc.rna_unit=="ng/nl")] = ((rna_conc.rna_conc)/rna_conc.rna_weight)*1000
rna_conc.rna_um[(rna_conc.rna_unit=='ug/ml')] = ((rna_conc.rna_conc)/(rna_conc.rna_weight*1000))*1000

conc =pd.concat([protein_conc.uM, rna_conc.rna_um], axis="columns")

df_rnaphasep1_1["rna_conc_uM"] = rna_conc.rna_um
df_rnaphasep1_1["protein_conc_uM"] = conc.uM
#uM -> log(uM)
df_rnaphasep1_1["rna_conc_log"] = np.log(rna_conc.rna_um.values)
df_rnaphasep1_1["protein_conc_log"] = np.log(protein_conc.uM.values)


#temperature->摂氏・室温は25とする。数値だけのカラムを作成
df_rnaphasep1_1.temperature = df_rnaphasep1_1.temperature.str.replace("RT", "25")
df_rnaphasep1_1["temp"] = df_rnaphasep1_1.temperature.str.extract("(\d*)")
#pH -> float
#df_rnaphasep1_1.pH[~df_rnaphasep1_1.pH.str.contains("-")] = df_rnaphasep1_1.pH[~df_rnaphasep1_1.pH.str.contains("-")] .astype("float")
#-無限は外す
df_rnaphasep1_2 = df_rnaphasep1_1[(df_rnaphasep1_1.rna_conc_log != (-np.inf))]
df_rnaphasep1_2 = df_rnaphasep1_2[~df_rnaphasep1_1.pH.str.contains("-", na = True)]

#morphology_add(追加項目)が書き忘れているやつをmorphology(元の項目)で埋める
df_rnaphasep1_1.morphology_add = df_rnaphasep1_1.morphology_add.replace('-', np.nan)
df_rnaphasep1_1.morphology_add= df_rnaphasep1_1.morphology_add.fillna(df_rnaphasep1_1.morphology)
df_rnaphasep1_1 = df_rnaphasep1_1[~df_rnaphasep1_1.morphology_add.str.contains(",")]

#df_rnaphasep1_2.to_csv("/home/chin/rnaphasep_0802/rnap_0802_prepro_1.csv")
#np.infは後でdrop,
def seq_to_record(rna_seq, i, x):    
    seq_r2=SeqRecord(Seq(rna_seq))
    #idが被らないようもとdfのインデックスタンパク質名の先頭に
    a = str(x) # + str(i)
    seq_r2.id=f"{a}"
    return seq_r2
records=[]
for (seq, index, name) in zip(df_rnaphasep1_1.rna_sequence.values,range(df_rnaphasep1_1.shape[0]), df_rnaphasep1_1.index):
    a=seq_to_record(str(seq), index, name)
    records.append(a)

os.makedirs("./preprocessing_results/", exist_ok=True)
SeqIO.write(records, "./data/rna_seq.fasta", "fasta")  

df_rnaphasep1_1.drop("protein_weight", axis = "columns").to_csv("./preprocessing_results/rnap_preprocessing_1.csv", index=False)
# #histgram作成
# hist_cols = ["rna_conc_uM", "protein_conc_uM", "rna_conc_log", "protein_conc_log", "pH", "temp"]
# for col in hist_cols:
#     fig = plt.figure()
#     df_rnaphasep1_2[col].hist()
#     plt.title(col, fontsize = 20)
#     plt.xlabel(col, fontsize = 15)
#     plt.ylabel("frequency", fontsize = 15)


