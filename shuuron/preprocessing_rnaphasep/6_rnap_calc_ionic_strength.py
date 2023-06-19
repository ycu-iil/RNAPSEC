import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis, molecular_weight
import numpy as np
import codecs
import os
import math
import pyEQL

def list_to_df(target_list, target_idx):
    target_dict={}
    for i in range(len(target_list)):

        aa=str(target_list[i].magnitude)
        target_dict[target_idx[i]]=aa

    df_b=pd.DataFrame(target_dict,index=target_dict.keys())
    df_target_list=pd.DataFrame(df_b.T.iloc[:,0])
    df_target_list=df_target_list.rename(columns={df_target_list.columns[0]:0})
    return df_target_list
def main():

    df_salt = pd.read_csv("./prepro_results/rnap_salt_conc_target.csv").set_index("Unnamed: 0")
    df = pd.read_csv("./prepro_results/rnap_prepro_ex_3.csv")
    df_salt= df.loc[:, df_salt.columns]
    df_salt["pH"] =df.ph.copy()

    df_salt1 = df_salt[(df_salt.salt_1_conc.isna()) & (~df_salt.salt_0_conc.isna())].iloc[:, 0:3]
    df_salt1["pH"] = df.ph.copy()
    df_salt1["salt_0_name"] = df_salt1.salt_0_name.str.split(" ", expand = True)[1]

    #pyEQLのpH：numじゃないと無理
    #1価の塩: ' NaCl', ' KCl',
    salt_ionic_strength_ion=[]
    salt1_index=[]
    for i in df_salt1.index:
        a=df_salt1.salt_0_conc[i]+" "+df_salt1.salt_0_unit[i]
        b = df_salt1.salt_0_name[i]
        
        e=(float(df_salt1.salt_0_conc[i])*2) #2価なので、濃度2倍に設定
        f=str(e)+" "+df_salt1.salt_0_unit[i]
        if b=="NaCl" :
            #pHなし
            if pd.isnull(df_salt1.pH.loc[i]):
            #df_salt1.pH.loc[i].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]])
                salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                salt1_index.append(i)
            #pHあり
            else:
                c=math.floor(df_salt1.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]],pH=c)
                salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                salt1_index.append(i)
        elif b=="KCl":
            #pHなし
            if pd.isnull(df_salt1.pH.loc[i]):
            #df_salt1.pH.loc[i].isna():
                salt_a=pyEQL.Solution([["K+",a],["Cl-",a]])
                salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                salt1_index.append(i)
            #pHあり
            else:
                c=math.floor(df_salt1.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["K+",a],["Cl-",a]], pH=c)
                salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                salt1_index.append(i)
        elif b=="MgCl2":
            #pHなし
            if pd.isnull(df_salt1.pH.loc[i]):
            #df_salt1.pH.loc[i].isna():
                salt_a=pyEQL.Solution([["Mg2+",d],["Cl-",f]])
                salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                salt1_index.append(i)
            #pHあり
            else:
                c=math.floor(df_salt1.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Mg2+",d],["Cl-",f]], pH=c)
                salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                salt1_index.append(i)
        elif b=="ZnCl2":
            #pHなし
            if pd.isnull(df_salt1.pH.loc[i]):
            #df_salt1.pH.loc[i].isna():
                salt_a=pyEQL.Solution([["Zn2+", d], ["Cl-",f]],pH=c)
                salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                salt1_index.append(i)
            #pHあり
            else:
                c=math.floor(df_salt1.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Zn2+", d], ["Cl-",f]], pH=c)
                salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                salt1_index.append(i)
                
        elif b=="MgSO4":
                #pHなし
                if pd.isnull(df_salt1.pH.loc[i]):
                #df_salt1.pH.loc[i].isna():
                    salt_a=pyEQL.Solution([["Mg2+",a],["SO42-",a]])
                    salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                    salt1_index.append(i)
                #pHあり
                else:
                    c=math.floor(df_salt1.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Mg2+",a],["SO42-",a]], pH=c)
                    salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
                    salt1_index.append(i)
        elif b == "Zn2+":
            c=math.floor(df_salt1.pH.loc[i]) 
            salt_a=pyEQL.Solution([["Zn2+",a],],pH=c)
            salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
            salt1_index.append(i)
            
        #塩が計算できないやつ（尿素？とか）
        else:
            salt_ionic_strength_ion.append(None)
            salt1_index.append(i)

    salt1_dict={}
    for i in range(len(salt_ionic_strength_ion)):
        if salt_ionic_strength_ion[i]!=None:
            #print(salt1_index[i])
            a=str(salt_ionic_strength_ion[i].magnitude)
            salt1_dict[salt1_index[i]]=a
        else:
            salt1_dict[salt1_index[i]]=None
    df_a=pd.DataFrame(salt1_dict,index=salt1_dict.keys())
    df_salt1_is=pd.DataFrame(df_a.T.iloc[:,0])
    df_salt1_is=df_salt1_is.rename(columns={df_salt1_is.columns[0]:0})

    df_salt2 = df_salt[(df_salt.salt_0_conc.notna()) & (df_salt.salt_1_conc.notna()) & (df_salt.salt_2_conc.isna())]
    df_salt2 = df_salt2.loc[:, ['salt_0_conc', 'salt_0_unit', 'salt_0_name', 'salt_1_conc',
        'salt_1_unit', 'salt_1_name', "pH"]]


    #####塩2
    conc1 = df_salt2.salt_0_conc
    unit1 = df_salt2.salt_0_unit
    conc2 = df_salt2.salt_1_conc
    unit2 = df_salt2.salt_1_unit
    name1 = df_salt2.salt_0_name
    name2 = df_salt2.salt_0_name

    salt2_is = []
    salt2_index = []
    salt1_range_is=[]
    salt1_range_index=[]

    for i in df_salt2.index:
        #2価のイオンはname2の方にしかないです
        a=conc1.loc[i] +" "+unit1.loc[i]
        d=conc2.loc[i].astype("str")+" "+unit2.loc[i] #2価のイオンを含む
        e=(float(conc2.loc[i])*2) #2価なので、濃度2倍に設定
        f=str(e)+" "+unit2.loc[i]


        if name1.loc[i]=="NaCl" :
            
            if name2.loc[i] == "MgCl2" :
                #pHなし
                if pd.isnull(df_salt2.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f]])
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
                    print(i)
                #pHあり
                else:
                    c=math.floor(df_salt2.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f]],pH=c)
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
            elif name2.loc[i] == 'ZnCl2':
                #pHなし
                if pd.isnull(df_salt2.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Zn2+", d], ["Cl-",f]])
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
                #pHあり
                else:
                    c=math.floor(df_salt2.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Zn2+", d], ["Cl-",f]],pH=c)
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
            elif name2.loc[i] == 'KCl':
                if pd.isnull(df_salt2.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["K+", d], ["Cl-",d]])
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
                #pHあり
                else:
                    c=math.floor(df_salt2.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Na+",d],["Cl-",d]],pH=c)
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
                    
            elif name2.loc[i] == 'NaCl':
                if pd.isnull(df_salt2.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]])
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
                
                #pHあり
                else:
                    c=math.floor(df_salt2.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]],pH=c)
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
            else:
                salt2_is.append(salt_a.get_ionic_strength())

                salt2_index.append(i)
                
        elif name1.loc[i]=="KCl":
            if name2.loc[i] == "MgCl2" :
                #pHなし
                if pd.isnull(df_salt2.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f]])
                    salt2_is.append(salt_a.get_ionic_strength())
                    salt2_index.append(i)
                    print(i)
                #pHあり
                else:
                    c=math.floor(df_salt2.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f]],pH=c)
                    salt2_is.append(salt_a.get_ionic_strength())

                    salt2_index.append(i)
            elif name2.loc[i] == 'ZnCl2':
                #pHなし
                if pd.isnull(df_salt2.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Zn2+", d], ["Cl-",f]])
                    salt2_is.append(salt_a.get_ionic_strength())

                    salt2_index.append(i)
                #pHあり
                else:
                    c=math.floor(df_salt2.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Zn2+", d], ["Cl-",f]],pH=c)
                    salt2_is.append(salt_a.get_ionic_strength())

                    salt2_index.append(i)
            elif name2.loc[i] == 'KCl':
                if pd.isnull(df_salt2.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["K+", d], ["Cl-",d]])
                    salt2_is.append(salt_a.get_ionic_strength())

                    salt2_index.append(i)
                #pHあり
                else:
                    c=math.floor(df_salt2.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Na+",d],["Cl-",d]],pH=c)
                    salt2_is.append(salt_a.get_ionic_strength())

                    salt2_index.append(i)
                    
            elif name2.loc[i] == 'NaCl':
                if pd.isnull(df_salt2.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]])
                    salt2_is.append(salt_a.get_ionic_strength())

                    salt2_index.append(i)
            
                #pHあり
                else:
                    c=math.floor(df_salt2.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]],pH=c)
                    salt2_is.append(salt_a.get_ionic_strength())

                    salt2_index.append(i)
            
        #塩が計算できないやつ（尿素？とか）
        else:
            salt2_is.append(salt_a.get_ionic_strength())
            salt2_index.append(i)

    #リスト→辞書→pandas
    df_salt2_is = list_to_df(salt2_is, target_idx = salt2_index)
    ######塩3

    "3種塩"
    df_salt3 = df_salt[(df_salt.salt_2_conc.notna()) & (df_salt.salt_3_conc.isna())]
    df_salt3 = df_salt3.loc[:, ['salt_0_conc', 'salt_0_unit', 'salt_0_name', 'salt_1_conc',
        'salt_1_unit', 'salt_1_name', 'salt_2_conc', 'salt_2_unit',
        'salt_2_name', "pH"]]

    #塩の種類ごとのconc
    conc3_0, conc3_1, conc3_2 = df_salt3.loc[:, "salt_0_conc"], df_salt3.loc[:, "salt_1_conc"], df_salt3.loc[:, "salt_2_conc"]
    #塩の種類ごとの単位
    unit3_0, unit3_1, unit3_2 = df_salt3.loc[:, "salt_0_unit"], df_salt3.loc[:, "salt_1_unit"], df_salt3.loc[:, "salt_2_unit"]
    #塩の種類
    name3_0, name3_1, name3_2 = df_salt3.loc[:, "salt_0_name"], df_salt3.loc[:, "salt_1_name"], df_salt3.loc[:, "salt_2_name"]

    "計算"
    salt3_is = []
    salt3_index = []
    for i in df_salt3.index:
        #2価のイオンはname3_1の方にしかないです
        a=conc3_0.loc[i]+" "+unit3_0.loc[i]
        conc3_2_2 = (float(conc3_2.loc[i])*2)
        
        d=conc3_1.loc[i].astype("str")+" "+unit3_1.loc[i] #2価のイオンを含む
        e=(float(conc3_1.loc[i])*2) #2価なので、濃度2倍に設定
        
        f=str(e)+" "+unit3_1.loc[i]
        
        g = conc3_2.loc[i].astype("str") +" "+unit3_2.loc[i]
        
        h2 = (float(conc3_2.loc[i])*2)
        h=str(h2)+" "+unit3_2.loc[i]
        #h: 3コメの濃度のにヴァイ

        if name3_0.loc[i]=="NaCl" :
            
            if (name3_1.loc[i] == "MgCl2") & (name3_2.loc[i] == "ZnCl2"):
                #pHなし
                if pd.isnull(df_salt3.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f], ["Zn2+", g], ["Cl", h]]) #g: 濃度1倍, h: 濃度2倍
                    salt3_is.append(salt_a.get_ionic_strength())
                    salt3_index.append(i)
                    print(i)
                #pHあり
                else:
                    c=math.floor(df_salt3.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f], ["Zn2+", g], ["Cl", h]], pH= c) #g: 濃度1倍, h: 濃度2倍
                    salt3_is.append(salt_a.get_ionic_strength())
                    salt3_index.append(i)
            if (name3_1.loc[i] == "KCl") & (name3_2.loc[i] == "MgCl2"):
                #pHなし
                if pd.isnull(df_salt3.pH.loc[i]):
                #ph3.pH.loc[f2["index"].loc[i]].isna():
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a], ["K+", d], ["Cl-",d ],["Mg2+", g],["Cl-", h]]) #g: 濃度1倍, h: 濃度2倍
                    salt3_is.append(salt_a.get_ionic_strength())
                    salt3_index.append(i)
                    print(i)
                #pHあり
                else:
                    c=math.floor(df_salt3.pH.loc[i]) 
                    #print(i)
                    salt_a=pyEQL.Solution([["Na+",a],["Cl-",a], ["K+", d], ["Cl-", d],["Mg2+", g],["Cl-", h]]) #g: 濃度1倍, h: 濃度2倍
                    salt3_is.append(salt_a.get_ionic_strength())
                    salt3_index.append(i)
            else:
                print("w")
        #塩が計算できないやつ（尿素？とか）
        else:
            salt3_is.append(salt_a.get_ionic_strength())
            salt3_index.append(i)
    df_salt3_is = list_to_df(salt3_is, target_idx = salt3_index)

    #塩4→ KH2PO$: 3件, phosphate (pyEQL対応してない): 6件

    "3種塩"
    df_salt4 = df_salt[(df_salt.salt_2_name != "phosphate") & (df_salt.salt_3_conc.notna())]


    #塩の種類ごとのconc
    conc4_0, conc4_1, conc4_2 = df_salt4.loc[:, "salt_0_conc"], df_salt4.loc[:, "salt_1_conc"], df_salt4.loc[:, "salt_2_conc"]
    # 濃度、単位、塩種類名ごとに変数に入れる
    unit4_0, unit4_1, unit4_2 = df_salt4.loc[:, "salt_0_unit"], df_salt4.loc[:, "salt_1_unit"], df_salt4.loc[:, "salt_2_unit"]
    name4_0, name4_1, name4_2 = df_salt4.loc[:, "salt_0_name"], df_salt4.loc[:, "salt_1_name"], df_salt4.loc[:, "salt_2_name"]
    conc4_3, name4_3, unit4_3 = df_salt4.loc[:, "salt_3_conc"], df_salt4.loc[:, "salt_3_name"],  df_salt4.loc[:, "salt_3_unit"]

    #計算
    salt4_is = []
    salt4_index = []
    for i in df_salt4.index:
        str_conc_unit_0 =conc4_0[i] +" "+unit4_0[i]
        str_name_0 = name4_0[i]

        str_conc_unit_1=str(conc4_1[i]) +" "+unit4_1[i]
        str_name_1 = name4_1[i]

        str_conc_unit_2= str(conc4_2[i]) +" "+unit4_2[i]
        str_name_2 = name4_2[i]

        int_conc_double_3=(float(conc4_3[i])*2) #2価なので、濃度2倍に設定
        str_conc_double_3=str(int_conc_double_3)+" "+unit4_3[i]
        str_conc_unit_3 = str(conc4_3[i]) +" "+unit4_3[i]
        str_name_3 = name4_3[i]

        salt_a=pyEQL.Solution([["Na+",str_conc_unit_0],["Cl-",str_conc_unit_0],
                            ["K+", str_conc_unit_1], ["Cl-",str_conc_unit_1 ],
                            ["KH+", str_conc_unit_2], ["HPO4-", str_conc_unit_2],
                            ["Mg2+", str_conc_unit_3],["Cl-", str_conc_double_3]])
        salt4_is.append(salt_a.get_ionic_strength())
        salt4_index.append(i)
    df_salt4_is = list_to_df(salt4_is, target_idx = salt4_index)
    ######保存
    df_salt2_is = df_salt2_is.rename(columns = {35:0}) 
    df_is   = pd.concat([df_salt1_is, df_salt2_is, df_salt3_is, df_salt4_is], axis = "index").rename(columns = {0: "ionic_strength"})
    df = pd.concat([df, df_is], axis = "columns")
    df.to_csv("./prepro_results/rnap_calc_is_6.csv")
    pass

if __name__ == "__main__":
    main()
    