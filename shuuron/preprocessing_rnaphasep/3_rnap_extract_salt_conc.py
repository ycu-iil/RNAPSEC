import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis, molecular_weight
import numpy as np
import codecs
import os

######## salt concentration -> ionic strength
#単位→2種類 mM, uM
# 塩の種類→NaCl, KCl, MgCl2, CaCl2, ZnCl2, MgSO4, KH2PO4, phosphate
# 塩の種類数 → 1, 2, 3, 4
###↓表記のバリエーション
##塩の種類数に応じてカラムを横のばしにする
# 塩なし → '-' ->np.nan -> 0
#1 塩1, 1点表記: '100 mM NaCl', -> 100 | mM | NaCl
#2 塩1, 範囲表記 (6件) : '0-150 mM NaCl' -> 範囲計算 -> 75 | mM | NaCl
#3 塩2, OR 繋ぎ: '50mM NaCl or 150mM NaCl' -> データ数によっては落とす ->morphologyの時点で落ちてた
#4 塩2, 1点表記:'150 mM NaCl, 1 mM MgCl2' -> 150|mM|NaCl|1|mM|MgCl2
#5 塩2, 片方範囲表記: '50-200 mM NaCl, 5 mM MgCl2' -> 範囲の平均 -> 平均|mM|NaCl|5|mM|MgCl2
#6 塩３, 1点表記: '150 mM NaCl, 1 mM MgCl2, 4 mM ZnCl2' → 150|mM|NaCl|5|mM|MgCl2|4|mM|ZnCl2
#7 塩４, 1点表記: '15 mM NaCl, 130 mM KCl, 5 mM KH2PO4, 1.5 mM MgCl2' → 150|mM|NaCl|5|mM|MgCl2|4|
def main():
    os.makedirs("./prepro_results/", exist_ok=True)
    df_rnaphasep1_2 = pd.read_csv("./prepro_results/rnap_prepro_ex_2.csv", )
    df_rnaphasep1_2["salt_edit"] = df_rnaphasep1_2.salt_concentration.copy() #salt_conc編集用のカラム
    df_rnaphasep1_2.salt_edit[df_rnaphasep1_2.salt_edit == "-"] = 0 #saltを加えてない実験系として、0を代入
    df_rnaphasep1_2.salt_edit = df_rnaphasep1_2.salt_edit.str.replace("(150 mM \()", "") # '150 mM (37.5 mM\nKCl, 112.5 mM NaCl)'
    df_rnaphasep1_2.salt_edit = df_rnaphasep1_2.salt_edit.str.replace("(\))", "")
    df_rnaphasep1_2.salt_edit = df_rnaphasep1_2.salt_edit.str.replace("µ", "u")
    df_salt_split = df_rnaphasep1_2.salt_edit.str.split("(, |,)", expand = True)
    df_salt_target = df_salt_split.loc[:,[0, 2, 4, 6]]
    #塩の数ごとにフレーム分ける
    df_salt1 = df_salt_target[df_salt_target[2].isna()].loc[:, 0]
    df_salt2_0 = df_salt_target[(df_salt_target[6].isna()) & (df_salt_target[4].isna())&(df_salt_target[2].notna())].loc[:, [0, 2]]
    #", "で区切れなかったやつ(",  ")
    df_salt2_ex = df_rnaphasep1_2.salt_edit.str.split("(,  )", expand = True)[df_rnaphasep1_2.salt_edit.str.split("(,  )", expand = True)[1].notna()].loc[:, [0,2]]
    df_salt2_ = df_salt2_0[~df_salt2_0.index.isin(df_salt2_ex.index)]
    df_salt2 = pd.concat([df_salt2_, df_salt2_ex], axis = "index")
    df_salt3 = df_salt_target[(df_salt_target[6].isna()) & (df_salt_target[4].notna())].loc[:, [0, 2, 4]]
    df_salt4 = df_salt_target[(df_salt_target[6].notna())].loc[:, [0, 2, 4, 6]]

    def concat_salt(salt = df_salt2):
        salt_list = []
        for k, col_oya in enumerate(salt.columns):
            aa = salt[col_oya].str.split(" ", expand = True)
            for col_ko, col_name in enumerate(["conc", "unit", "name"]):
                aa = aa.rename(columns= {col_ko:f"{k}_{col_name}"})
            salt_list.append(aa)
        df_salt_concat = pd.concat(salt_list, axis = "columns")
        return df_salt_concat

    df_salt2_con = concat_salt()
    df_salt3_con= concat_salt(salt = df_salt3)
    df_salt4_con= concat_salt(salt = df_salt4)

    #1, 2
    unit_list = ["uM", "mM"]
    df_salt1_list = []
    for unit in unit_list:
        df_salt1_unit = df_salt1[df_salt1.str.contains(f"{unit}", na = False)]
        df_salt1_value = df_salt1_unit.str.split(f"{unit}", expand = True) #値と単位の間にスペースがないデータが混ざっているので、上の関数で処理できない→溶質濃度と同じやり方で取り出した
        df_salt1_value = df_salt1_value.rename(columns = {0: "0_conc", 1: "0_name"})
        df_salt1_value["0_unit"] = f"{unit}"
        df_salt1_value = df_salt1_value.loc[:, ["0_conc","0_unit", "0_name"]]
        # 1塩で範囲表記を平均化
        df_salt1_pt = df_salt1_value[~df_salt1_value["0_conc"].str.contains("-", na = True)]
        df_salt1_range = df_salt1_value[df_salt1_value["0_conc"].str.contains("-", na = False)]

        df_salt1_range_0 =df_salt1_range["0_conc"].str.split("-", expand = True).astype("float")
        df_salt1_range_0["0_conc"] =df_salt1_range_0.mean(axis = "columns") 
        df_salt1_range ["0_conc"] = df_salt1_range_0["0_conc"].copy()
        #1塩の一点濃度と合体
        df_salt1_unit_concat= pd.concat([df_salt1_pt, df_salt1_range], axis = "index")
        df_salt1_list.append(df_salt1_unit_concat)
    df_salt1_conc = pd.concat(df_salt1_list, axis = "index")
    #1塩の一点濃度と合体
    

    #範囲表記の計算->1点濃度と結合 (塩１の平均化処理を関数化)
    def calc_range(df_input_value, salt_num):
        df_input_value = df_input_value.loc[:, [f"{salt_num}_conc",f"{salt_num}_unit", f"{salt_num}_name"]]
        df_input_pt = df_input_value[~df_input_value[f"{salt_num}_conc"].str.contains("-", na = True)]
        df_input_range = df_input_value[df_input_value[f"{salt_num}_conc"].str.contains("-", na = False)] #範囲表記だけ抜き出して計算後、1点濃度と合体
        # "-"で区切って平均値を計算
        df_input_range_0 =df_input_range[f"{salt_num}_conc"].str.split("-", expand = True).astype("float")
        df_input_range_0[f"{salt_num}_conc"] =df_input_range_0.mean(axis = "columns") #平均化
        df_input_range [f"{salt_num}_conc"] = df_input_range_0[f"{salt_num}_conc"].copy()
        df_input_conc= pd.concat([df_input_pt, df_input_range], axis = "index")
        return df_input_conc

    #塩2
    df_salt2_0 = calc_range(df_input_value = df_salt2_con.loc[:, ["0_conc", "0_unit", "0_name"]], salt_num = 0)
    df_salt2_1 = calc_range(df_input_value = df_salt2_con.loc[:, ["1_conc", "1_unit", "1_name"]], salt_num = 1)
    df_salt2_conc = pd.concat([df_salt2_0, df_salt2_1], axis = "columns")
    # 塩3
    df_salt3_0 = calc_range(df_input_value = df_salt3_con.loc[:, ["0_conc", "0_unit", "0_name"]], salt_num = 0)
    df_salt3_1 = calc_range(df_input_value = df_salt3_con.loc[:, ["1_conc", "1_unit", "1_name"]], salt_num = 1)
    df_salt3_2 = calc_range(df_input_value = df_salt3_con.loc[:, ["2_conc", "2_unit", "2_name"]], salt_num = 2)
    df_salt3_conc = pd.concat([df_salt3_0, df_salt3_1, df_salt3_2], axis = "columns")
    #塩4
    df_salt4_0 = calc_range(df_input_value = df_salt4_con.loc[:, ["0_conc", "0_unit", "0_name"]], salt_num = 0)
    df_salt4_1 = calc_range(df_input_value = df_salt4_con.loc[:, ["1_conc", "1_unit", "1_name"]], salt_num = 1)
    df_salt4_2 = calc_range(df_input_value = df_salt4_con.loc[:, ["2_conc", "2_unit", "2_name"]], salt_num = 2)
    df_salt4_3 = calc_range(df_input_value = df_salt4_con.loc[:, ["3_conc", "3_unit", "3_name"]], salt_num = 3)
    df_salt4_conc = pd.concat([df_salt4_0, df_salt4_1, df_salt4_2, df_salt4_3], axis = "columns")
    #塩1-4を合体
    df_salt_concat = pd.concat([df_salt1_conc, df_salt2_conc, df_salt3_conc, df_salt4_conc], axis = "index").sort_index()

    #カラム名変更、rnaphasepと結合
    for col in df_salt_concat.columns:
        df_salt_concat = df_salt_concat.rename(columns = {col: f"salt_{col}"})
    
    df_salt_concat.to_csv("./prepro_results/rnap_salt_conc_target.csv", index=True) #別スクリプトでイオン強度を計算
    df_rnaphasep1_3 = pd.concat([df_rnaphasep1_2, df_salt_concat], axis = "columns")
    df_rnaphasep1_3 = df_rnaphasep1_3.rename(columns = {"Unnamed: 0": "ini_idx"})
    df_rnaphasep1_3.to_csv("./prepro_results/rnap_prepro_ex_3.csv", index=True)

    pass

if __name__ == "__main__":
    main()

