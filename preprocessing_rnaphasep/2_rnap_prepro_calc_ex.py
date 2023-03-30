import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis, molecular_weight
import numpy as np
import codecs
import os
#タンパク質・RNA濃度の単位を統一
#pH, 温度の数値抽出

def main():
    os.makedirs("./prepro_results/", exist_ok=True)
    df_rnaphasep1 = pd.read_csv("./prepro_results/rnap_extract_conc_1.csv", index_col = False)
    df_rnaphasep1["aa"] = np.nan
    df_rnaphasep1["aa"][~df_rnaphasep1.protein_sequence.str.contains("\|")] = df_rnaphasep1.protein_sequence
    df_rnaphasep1["aa"][df_rnaphasep1.protein_sequence.str.contains("\|")] = df_rnaphasep1.protein_sequence.str.split("\n", 1, expand = True)[1].str.replace("\n|\||;", "")
    df_rnaphasep1["aa"] = df_rnaphasep1["aa"].str.replace("(^[A-Z])","",  regex = False)
    df_rnaphasep1["aa"] = df_rnaphasep1["aa"].str.replace(">spQ99496RING2_HUMAN E3 ubiquitin-protein ligase RING2 OS=Homo sapiens OX=9606 GN=RNF2 PE=1 SV=1", "") #fusion protein
    df_rnaphasep1 = df_rnaphasep1[df_rnaphasep1.aa.str.contains("[A-Z][A-Z]", na = False)]
    df_rnaphasep1 = df_rnaphasep1[df_rnaphasep1.rna_sequence.str.contains("([A-Z])", na = False)]
    df_rnaphasep1.rna_sequence = df_rnaphasep1.rna_sequence.str.replace(";\|-", "")
    #最初に全部floatに変換する
    df_rnaphasep1.protein_conc = df_rnaphasep1.protein_conc.astype("float")
    df_rnaphasep1.rna_conc = df_rnaphasep1.rna_conc.astype("float")
    #濃度欠損のデータをはずす
    #na, 0 を含むデータをはずす

    #####濃度の単位換算
    ##protein → ug, ul, nM, uM. mM, ng/ul, mg/ml
    df_rnaphasep1 = df_rnaphasep1[(df_rnaphasep1.rna_conc.notna()) & (df_rnaphasep1.protein_conc.notna())] #
    df_rnaphasep1 = df_rnaphasep1[(df_rnaphasep1.rna_conc != 0) & (df_rnaphasep1.protein_conc != 0)]
    df_rnaphasep1 = df_rnaphasep1[(df_rnaphasep1.rna_unit != "ug") & (df_rnaphasep1.protein_unit != "ug")]
    df_rnaphasep1 = df_rnaphasep1[(df_rnaphasep1.rna_unit != "ul") & (df_rnaphasep1.protein_unit != "ul")]
    df_rnaphasep1["protein_weight"]="?"
    for i in df_rnaphasep1.index:
        df_rnaphasep1.loc.__setitem__((i,"protein_weight"),  ("%0.2f" % molecular_weight(
            f"{df_rnaphasep1.aa.loc[i]}", "protein")))
    #単位の表記統一 (μ, 文字化け(元ファイルはμ) → u)
    df_rnaphasep1.rna_unit =df_rnaphasep1.rna_unit.str.replace('ﾎｼ|u|u', "u")
    df_rnaphasep1.protein_unit =df_rnaphasep1.protein_unit.str.replace('ﾎｼ|u|µ|_', "u")
    df_rnaphasep1.protein_unit =df_rnaphasep1.protein_unit.str.replace('_', "u")
    df_rnaphasep1.protein_unit =df_rnaphasep1.protein_unit.str.replace('um', "uM")
    #タンパク質濃度の単位をuMに統一
    protein_conc = df_rnaphasep1.loc[:, ['protein_conc','protein_unit',"protein_weight"]]
    protein_conc.protein_unit = protein_conc.protein_unit.str.replace('ﾎｼ|u|µ|_', "u")
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
    df_rnaphasep1.rna_unit =df_rnaphasep1.rna_unit.str.replace('ﾎｼ|u|µ|_', "u") #消すと全落ちする
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

    ###############temperature->摂氏・室温は25とする。数値だけのカラムを作成
    df_rnaphasep1_1["temp"] = df_rnaphasep1_1.temperature.str.replace("RT", "25")
    df_rnaphasep1_1["temp"] = df_rnaphasep1_1.temp.str.extract("(\d*)")

    #############pH
    c1 = df_rnaphasep1_1["buffer"].str.extract('(?P<pH>pH [0-9].[0-9]|pH [0-9]|pH=[0-9]|pH=[0-9]|pH [0-9].[0-9][0-9]|pH [0-9].[0-9][0-9]|pH [0-9]-[0-9][0-9]|pH [0-9]−[0-9]|pH [0-9].[0-9]−[0-9]|pH [0-9].[0-9]-[0-9]|pH [0-9].[0-9][0-9]-[0-9]|pH＜[0-9]|pH <[0-9]|pH[0-9].[0-9]|l (.+), 2|pH \((.+)\)|pH\((.+)\)|pH[0-9]|pH_(.+)\)|pH_(.+),|pH =(.+)\)|pH at (.+) w)',expand=True)
    c1["pH"][df_rnaphasep1_1["buffer"].str.contains("pH NA")]="Pass"
    c1["pH"][df_rnaphasep1_1["buffer"].str.contains("buffer|water") & c1.pH.isnull()]=7


    c2=pd.DataFrame(c1["pH"])
    c4=c2.pH.str.split("pH|_|l",1, expand=True)

    c5=pd.DataFrame(c4[1].str.split("-|=|−", expand=True))
    c4[1]=c4[1].str.replace("＜","<")

    #数値だけ、pHo-oは-より前までしか取れてないのであとで

    c4[1][c4[1].str.contains(", 2|water",na=False)]=7.4
    c4[1][df_rnaphasep1_1.buffer.str.contains("water",na=False)]=7.4
    df_rnaphasep1_1["ph"]=c4[1]

    df_rnaphasep1_2 = df_rnaphasep1_1.dropna()
    df_rnaphasep1_2 = df_rnaphasep1_2[(df_rnaphasep1_2.temp!="") & (df_rnaphasep1_2.ph != "")]


    ######## morphology
    ### morphologyの新しいカラムを作成。複数形態・結果なしを欠損値に、結果１つのみならそのまま
    #1 "liquid", "solute", "gel", "solid" → そのまま新しいカラムにコピー
    #2 "-" → 欠損値
    #3 "solute, α" → "α"として新しいカラムに記録
    #4 "solide/liquid/gel, solide/liquid/gel" → 欠損値

    #1
    df_rnaphasep1_2["morphology_add"] = df_rnaphasep1_2.morphology.copy()
    #2
    df_rnaphasep1_2.morphology_add[df_rnaphasep1_2.morphology_add.str.contains("-", na = True)] = np.nan
    #3
    df_rnaphasep1_2.morphology_add[(df_rnaphasep1_2.morphology_add.str.contains("solute", na = False)) & 
                                (df_rnaphasep1_2.morphology_add.str.contains("(liquid|gel|solid)", na = False))] = df_rnaphasep1_2.morphology_add.str.replace("solute, ", "")
    df_rnaphasep1_2.morphology_add[(df_rnaphasep1_2.morphology_add.str.contains("solute", na = False)) & 
                                (df_rnaphasep1_2.morphology_add.str.contains("(liquid|gel|solid)", na = False))] = df_rnaphasep1_2.morphology_add.str.replace(", solute", "")
    #4
    df_rnaphasep1_2.morphology_add[(df_rnaphasep1_2.morphology_add.str.contains(",", na = False)) & 
                                (df_rnaphasep1_2.morphology_add.str.contains("(liquid|gel|solid)", na = False))] = np.nan
    ######アミノ酸配列とRNA配列をfasta形式に保存
    def seq_to_record(rna_seq, i, x):    
        seq_r2=SeqRecord(Seq(rna_seq))
        #idが被らないようもとdfのインデックスタンパク質名の先頭に
        a = str(x) # + str(i)
        seq_r2.id=f"{a}"
        return seq_r2
    records=[]
    for (seq, index, name) in zip(df_rnaphasep1_2.rna_sequence.values,range(df_rnaphasep1_2.shape[0]), df_rnaphasep1_2.index):
        a=seq_to_record(str(seq), index, name)
        records.append(a)
    SeqIO.write(records, "./prepro_results/rnap_rnaseq.fasta", "fasta")  
    os.makedirs("./mf_result/", exist_ok=True)  #RNA配列→記述子の計算結果保存用ディレクトリ作成
    ##前処理後のデータフレームを保存
    df_rnaphasep1_2.to_csv("./prepro_results/rnap_prepro_ex_2.csv", index = True)
    pass

if __name__ == "__main__":
    main()
