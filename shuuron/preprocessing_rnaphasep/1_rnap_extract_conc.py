import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
#濃度を抽出
#RNAPhaSepの"solute_concentration"からタンパク質濃度とRNA濃度の値を取り出して、タンパク質とRNAでそれぞれ"濃度の値"と"濃度の単位"のカラムを作成する

def main():
    os.makedirs("./prepro_results/", exist_ok=True)
    df_all_data = pd.read_excel("./All data.xlsx", engine = "openpyxl") #RNAPhaSep(HP)からダウンロードしたファイルの読み込み
    df_all_data = df_all_data.drop(df_all_data.columns[df_all_data.columns.str.contains("Unnamed")], axis = "columns")
    df_all_data = df_all_data.rename(columns = {"index": "ini_idx"}) #基準のカラム
    df = df_all_data[df_all_data.components_type == "RNA + protein"] #タンパク質とRNAの種類が1種類ずつ
    df = df[df.solute_concentration.str.contains(";", na = False)] #複数分子の濃度が濃度のカラムに記載されている

    "数値と単位と分指名に区切る"
    ######################solute_concentrationの左側に記載されている分子から処理
    df_split_1 = df.solute_concentration.str.split(";\|", expand = True) #1カラムに2分子以上入っていた者を、一個ずつにカラムを分ける
    df_split_1 =  df_split_1[~df_split_1[2].str.contains("([0-9])", na = False)]

    ######右側の範囲表記を計算
    def calc_range(side = 1): #side: 右側 = 1, 左側 = 0
    #solute_concの左側の分子の数値単位分子名のカラム
            #'136 µM [PTBP1]',
            #≥125 nM [PTBP1]-[FUS IDR]'
            #1.8,18,180 nM[SARS-CoV-2]
            #'0.25-5µM[mCherry-N protein]'
            #'10\u2009µM [SARS-CoV-2 N protein]
            #'0.3nM-7.5 nM'
        df_split_1_0 = df_split_1[side].str.split("(\[)", expand = True) #数値単位 | 分指名にわける #右側分子指定
            #	136 µM	[	PTBP1]	None	None
        #範囲含むのと含まないので別の処理をする
        #範囲表記 ->92件
        df_split_1_0_range = df_split_1_0[df_split_1_0[0].str.contains("(-[0-9]|[<,>,~])", na = False)]
        df_split_1_0_range_value = df_split_1_0[df_split_1_0[0].str.contains("(-[0-9]|[<,>,~])", na = False)][0] # 数値と単位, たまに分子名： 単位の種類：µM, nM, µg/µL, mg/ml, ng/µl , 
        df_split_1_0_range_value = df_split_1_0_range_value.str.replace("nM-", "-") #0.3nM-7.5 nM → 0.3-0.7 nM
        unit_list = ["µM", "nM","mM", "µg/µL", "mg/ml", "ng/µl", "mg/mL", "ng/µL", "µm", "µg/mL", ] 
        #[範囲数値];[単位]
        range_list = []
        unit_dict = {}
        for unit in unit_list: #それぞれの単位を含むデータ群ごとのdfを作成し、結合
            print(unit)
            if df_split_1_0_range_value[df_split_1_0_range_value.str.contains(unit, na = False)].shape[0] >0 :
                df_split_1_0_range_value_0 = df_split_1_0_range_value.str.split(unit, expand = True)
                df_range = pd.DataFrame(df_split_1_0_range_value_0[df_split_1_0_range_value_0[1].notna()][0])
                df_range["unit"] = unit
                df_range["name"] = pd.DataFrame(df_split_1_0_range_value_0[df_split_1_0_range_value_0[1].notna()][1])
                range_list.append(df_range)
            else:
                continue
        df_range_right = pd.concat(range_list) #shape = 92, 0
        df_range_right["name"][(df_range_right["name"] == " ") |(df_range_right["name"] == "")] = np.nan
        # df_range_right["name"] = df_split_1_0_range[2].str.replace("(\])", "") #分子名のカラムを追加→左側区切れないのも多々ありい
        df_range_right["name"] = df_range_right["name"].fillna(df_split_1_0_range[2].str.replace("(\])", "")) #分子名のカラムを追加→左側区切れないのも多々ありい
        df_range_right = df_range_right.rename(columns={0:"value"})

        df_range_right["value"] = df_range_right["value"].str.replace("<", "0-") 
        df_range_right["value"] = df_range_right["value"].str.replace("~", "-")
        df_range_right["value"] = df_range_right["value"].str.replace("≤", "0-") 
        df_range_right2 = df_range_right[~df_range_right.value.str.contains(">|≥", na = True)]#>は2データしかないので使わない→左側の分子で範囲表記になってるデータ数: 90件
        #はに表記のデータの平均値を計算
        df_range_right2["value"] = df_range_right2.value.str.replace("(-|<)", ",")
        df_range_right_avg = df_range_right2.value.str.split(",", expand = True)
        for col in df_range_right_avg.columns:
            df_range_right_avg[col] = df_range_right_avg[col].astype("float")
        df_range_right_avg["avg"] = df_range_right_avg.mean(axis ="columns")
        print(df_range_right_avg.avg.isna().sum()) # =0なら処理忘れなし
        df_range_right_comp = df_range_right2.copy()
        df_range_right_comp["conc"] = df_range_right_avg["avg"] #平均化後の値は[conc]
        if side == 1:
            side_str = "right"
        else:
            side_str = "left"        
        for col in df_range_right_comp.columns:
            df_range_right_comp = df_range_right_comp.rename(columns = {col: f"{col}_{side_str}"})

        return df_range_right_comp

    df_range_right_comp = calc_range(side = 1)
    df_range_left_comp =  calc_range(side = 0)
    #左右合体
    df_range_comp= pd.concat([df_range_right_comp, df_range_left_comp], axis = "columns")
    df_range_comp= df_range_comp.drop(df_range_comp.columns[df_range_comp.columns.str.contains("value")], axis = "columns")
    df_range_comp= df_range_comp.loc[:, ['conc_right','unit_right', 'name_right','conc_left', 'unit_left', 'name_left']]

    ################1点濃度
    def calc_pt(side = 0):
        df_split_1_0 = df_split_1[side].str.split("(\[)", expand = True) #数値単位 | 分指名にわける #右側分子指定
        df_split_1_0_pt = df_split_1_0[~df_split_1_0[0].str.contains("(-[0-9]|[<,>,~,≥,≤]|[0-9]/[0-9])", na = False)] #点濃度だけに絞る
        df_split_1_0_pt_value = df_split_1_0_pt[0] #値と単位だけの列を選択
        unit_list = ["µg", "µl", "µM", "nM","mM", "µg/µL", "mg/ml", "ng/µl", "mg/mL", "ng/µL", "µm", "µg/mL", "µg/ml",  ] 
            #[範囲数値];[単位]
        range_list = []

        for unit in unit_list: #それぞれの単位を含むデータ群ごとのdfを作成し、結合
            print(unit)
            if df_split_1_0_pt_value[df_split_1_0_pt_value.str.contains(unit, na = False)].shape[0] >0 :
                df_unit = df_split_1_0_pt_value[df_split_1_0_pt_value.str.contains(unit, na = False)]
                df_split_1_0_pt_value_0 = df_unit.str.split(unit, expand = True) #値とその他にカラムをに分ける
                df_range = pd.DataFrame(df_split_1_0_pt_value_0[df_split_1_0_pt_value_0[1].notna()][0]) #値が入っているデータを選択(単位がこのループ回の行を選択)
                df_range["unit"] = unit
                df_range["name"] = pd.DataFrame(df_split_1_0_pt_value_0[df_split_1_0_pt_value_0[1].notna()][1]) #分子名が[]で囲まれてないのは先に入れとく
                range_list.append(df_range)
            else:
                continue
        df_pt = pd.concat(range_list, axis = "index")
        df_pt = df_pt[~df_pt.index.duplicated(keep = "last")].sort_index()
        df_pt["name"][(df_pt["name"] == " ") |(df_pt["name"] == "")] = np.nan #分子名：上のforで名前がなかった分がスペースで埋められていたので、nanに変換、後でfillnaで埋める
        df_pt["name"] = df_pt["name"].fillna(df_split_1_0_pt[2].str.replace("(\])", "")) #分子名のカラムを追加→左側区切れないのも多々ありい
        df_pt = df_pt.rename(columns={0:"value"})
        df_pt = df_pt[~df_pt.value.str.contains("([a-zA-Z])", na = True)] #変なの
        df_pt["conc"] = df_pt.value.astype("float")

        if side == 1:
            side_str = "right"
        else:
            side_str = "left"        
        for col in df_pt.columns:
            df_pt = df_pt.rename(columns = {col: f"{col}_{side_str}"})
        return df_pt

    df_pt_left_comp = calc_pt(side = 0)
    df_pt_right_comp = calc_pt(side = 1)

    df_pt_comp = pd.concat([df_pt_right_comp, df_pt_left_comp], axis = "columns")
    df_pt_comp = df_pt_comp.drop(df_pt_comp.columns[df_pt_comp.columns.str.contains("value")], axis = "columns")
    df_pt_comp = df_pt_comp.loc[:, ['conc_right','unit_right', 'name_right','conc_left', 
                                'unit_left', 'name_left']]

    ############タンパク質とRNAの仕分け
    df_pt_concat = df_pt_comp.fillna(df_range_comp).dropna() #片方1点濃度で片方範囲表記または両方1点濃度, 範囲表記を含むのは172件
    df_both_range_comp = df_range_comp.dropna() #両方とも範囲表記→30件

    df_sc= pd.concat([df_pt_concat, df_both_range_comp], axis = "index")  #459件、インデックス重複なし, 元の処理対象は533件


    #タンパク質名のリストを作成して、タンパク質かRNAかの仕分け
    protein_names = df.protein_name.unique()
    protein_name_list = protein_names.tolist()
    protein_name_list.append("rotein",)
    protein_name_list.append("IDR")
    protein_name_list.append("polyQ")
    protein_name_list.append("polyR")
    protein_name_list.append("SARS2-NP")
    #タンパク質名のリスト
    rna_name_list = ["RNA", "olyA", "olyU", "CGG", "olyC", "poly(rU)", "UTR"]
    #タンパク質かRNAかを0/1でラベリング
    df_sc_left = df_sc.loc[:, df_sc.columns[df_sc.columns.str.contains("left")]]
    df_sc_left["protein_left"] = np.nan
    for protein in protein_name_list:
        df_sc_left["protein_left"][df_sc_left.name_left.str.contains(protein)] = 1
    for rna in rna_name_list:
        df_sc_left["protein_left"][df_sc_left.name_left.str.contains(rna)] = 0

    df_sc_right = df_sc.loc[:, df_sc.columns[df_sc.columns.str.contains("right")]]
    df_sc_right["protein_right"] = np.nan
    for protein in protein_name_list:
        df_sc_right["protein_right"][df_sc_right.name_right.str.contains(protein)] = 1
    for rna in rna_name_list:
        df_sc_right["protein_right"][df_sc_right.name_right.str.contains(rna)] = 0

    df_sc_2 = pd.concat([df_sc_left, df_sc_right], axis = "columns")

    df_sc_comp = df_sc_2[(~df_sc_2.protein_left.isna()) | (~df_sc_2.protein_right.isna())] #どっちかは必ずラベリングされている
    #計算が難しい単位を省く(10件ほど)
    df_sc_comp_1 = df_sc_comp[df_sc_comp.unit_right != 'µg']
    df_sc_comp_1 = df_sc_comp_1[df_sc_comp_1.unit_right != 'µl']

    df_sc_comp_1.to_csv("./prepro_results/rnap_solute_concentration.csv", index = True)

    #######仕分け後→タンパク質のラベルを基にprotein_concのカラムとrna_concのカラムを作成
    df_protein_left = df_sc_comp_1[(df_sc_comp_1.protein_left == 1)| (df_sc_comp_1.protein_right == 0)].loc[:, ['conc_left', 'unit_left', 'name_left', "protein_left"]]
    df_protein_left = df_protein_left.rename(columns = {'conc_left': "protein_conc", 'unit_left': "protein_unit", 
                                                        'name_left': "protein_name", "protein_left":"is_protein"})
    df_protein_right = df_sc_comp_1[(df_sc_comp_1.protein_right == 1)| (df_sc_comp_1.protein_left == 0)].loc[:, ['conc_right', 'unit_right', 'name_right', "protein_right"]]
    df_protein_right = df_protein_right .rename(columns = {'conc_right': "protein_conc", 'unit_right': "protein_unit", 
                                                        'name_right': "protein_name", "protein_right":"is_protein"})
    df_protein_conc = pd.concat([df_protein_left, df_protein_right], axis = "index")
    df_protein_conc = df_protein_conc[~df_protein_conc.index.duplicated()]


    df_rna_left = df_sc_comp_1[(df_sc_comp_1.protein_left == 0) | (df_sc_comp_1.protein_right == 1)].loc[:, ['conc_left', 'unit_left', 'name_left', "protein_left"]]
    df_rna_left = df_rna_left.rename(columns = {'conc_left': "rna_conc", 'unit_left': "rna_unit", 
                                                        'name_left': "rna_name", "protein_left":"is_protein"})
    df_rna_right = df_sc_comp_1[(df_sc_comp_1.protein_right == 0)  | (df_sc_comp_1.protein_left == 1)].loc[:, ['conc_right', 'unit_right', 'name_right', "protein_right"]]
    df_rna_right = df_rna_right.rename(columns = {'conc_right': "rna_conc", 'unit_right': "rna_unit", 
                                                        'name_right': "rna_name", "protein_right":"is_protein"})

    df_rna_conc = pd.concat([df_rna_left, df_rna_right], axis = "index")
    df_rna_conc = df_rna_conc[~df_rna_conc.index.duplicated()]

    df_sc_comp_2 = pd.concat([df_rna_conc, df_protein_conc], axis = "columns")
    df_2 = pd.concat([df, df_sc_comp_2.drop("is_protein", axis = "columns")], axis = "columns") #RNAPhaSepの元と合体
    df_2.to_csv("./prepro_results/rnap_extract_conc_1.csv", index = False)

    
    pass

if __name__ == "__main__":
    main()
