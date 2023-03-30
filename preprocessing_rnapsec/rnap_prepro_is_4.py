import pyEQL
import pandas as pd
import numpy as np
import math


df_chin = pd.read_csv("./preprocessing_results/rnap_preprocessing_1.csv")
df_chin = df_chin.rename(columns = {"salt_unit/salt_name": "salt_unit_name"})
df_chin.pH[df_chin.pH =="-"] = 6.4
df_chin.pH = df_chin.pH.astype("float")
df_chin.salt_conc = df_chin.salt_conc.replace({"-":np.nan}, )
df_chin.salt_conc = df_chin.salt_conc.replace({None:np.nan})
df_chin.salt_unit_name = df_chin.salt_unit_name.str.replace("ﾎｼ|μ", "u")
df_chin.salt_unit_name[df_chin.salt_unit_name.str.contains("http", na = False)] = "mM/KCl, mM/MgCl2"
df_chin.salt_unit_name = df_chin.salt_unit_name.replace({None:np.nan})
df_salt = pd.DataFrame(df_chin.salt_conc)
df_salt["salt_unit"] = df_chin.salt_unit_name
df_salt["pH"] = df_chin.pH.astype("float")

#saltの種類が一種類のやつからやってみる
df_salt1 = df_salt[~df_salt.salt_conc.str.contains(",", na = True)]
salt1_unit = df_salt1.salt_unit.str.split("/", expand = True)
salt1_unit = salt1_unit.rename(columns = {0 : "unit", 1: "name"})
df_salt1_1 = pd.concat([df_salt1, salt1_unit], axis = "columns")

#pyEQLのpH：numじゃないと無理
salt_ionic_strength_ion=[]
salt1_index=[]
for i in df_salt1_1.index:
    a=df_salt1_1.salt_conc[i]+" "+df_salt1_1.unit[i]
    b = df_salt1_1.name[i]
    
    e=(float(df_salt1_1.salt_conc[i])*2) #2価なので、濃度2倍に設定
    f=str(e)+" "+df_salt1_1.unit[i]
    if b=="NaCl" :
        #pHなし
        if pd.isnull(df_salt1_1.pH.loc[i]):
        #df_salt1_1.pH.loc[i].isna():
            salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]])
            salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
            salt1_index.append(i)
        #pHあり
        else:
            c=math.floor(df_salt1_1.pH.loc[i]) 
            #print(i)
            salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]],pH=c)
            salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
            salt1_index.append(i)
    elif b=="KCl":
        #pHなし
        if pd.isnull(df_salt1_1.pH.loc[i]):
        #df_salt1_1.pH.loc[i].isna():
            salt_a=pyEQL.Solution([["K+",a],["Cl-",a]])
            salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
            salt1_index.append(i)
        #pHあり
        else:
            c=math.floor(df_salt1_1.pH.loc[i]) 
            #print(i)
            salt_a=pyEQL.Solution([["K+",a],["Cl-",a]], pH=c)
            salt_ionic_strength_ion.append(salt_a.get_ionic_strength())
            salt1_index.append(i)
            
    elif b == "Zn2+":
        c=math.floor(df_salt1_1.pH.loc[i]) 
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

##2種塩
salt_conc = df_salt.salt_conc.str.split(",", expand = True)
salt_unit = df_salt.salt_unit.str.split(",", expand = True)

salt_conc2 = salt_conc[salt_conc[2].isna() & salt_conc[1].notna()]
salt_unit2 = salt_unit[salt_unit.index.isin(salt_conc2.index)]

unit1 = salt_unit2[0].str.split("/", expand = True)[0]
name1 = salt_unit2[0].str.split("/", expand = True)[1]
unit2 = salt_unit2[1].str.split("/", expand = True)[0]
name2 = salt_unit2[1].str.split("/", expand = True)[1]
conc1 = salt_conc2[0]
conc2 = salt_conc2[1]

#conc1, unit1, name1
#conc2, unit2, name2

salt2_is = []
salt2_index = []
salt1_range_is=[]
salt1_range_index=[]

for i in salt_unit2.index:
    #2価のイオンはname2の方にしかないです
    a=conc1.loc[i]+" "+unit1.loc[i]
    d=conc2.loc[i]+" "+unit2.loc[i] #2価のイオンを含む
    e=(float(conc2.loc[i])*2) #2価なので、濃度2倍に設定
    f=str(e)+" "+unit2.loc[i]


    if name1.loc[i]=="NaCl" :
        
        if name2.loc[i] == "MgCl2" :
            #pHなし
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f]])
                salt2_is.append(salt_a.get_ionic_strength())
                salt2_index.append(i)
                print(i)
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f]],pH=c)
                salt2_is.append(salt_a.get_ionic_strength())
                salt2_index.append(i)
        elif name2.loc[i] == 'ZnCl2':
            #pHなし
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Zn2+", d], ["Cl-",f]])
                salt2_is.append(salt_a.get_ionic_strength())
                salt2_index.append(i)
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Zn2+", d], ["Cl-",f]],pH=c)
                salt2_is.append(salt_a.get_ionic_strength())
                salt2_index.append(i)
        elif name2.loc[i] == 'KCl':
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["K+", d], ["Cl-",d]])
                salt2_is.append(salt_a.get_ionic_strength())
                salt2_index.append(i)
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Na+",d],["Cl-",d]],pH=c)
                salt2_is.append(salt_a.get_ionic_strength())
                salt2_index.append(i)
                
        elif name2.loc[i] == 'NaCl':
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]])
                salt2_is.append(salt_a.get_ionic_strength())
                salt2_index.append(i)
            
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
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
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f]])
                salt2_is.append(salt_a.get_ionic_strength())
                salt2_index.append(i)
                print(i)
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f]],pH=c)
                salt2_is.append(salt_a.get_ionic_strength())

                salt2_index.append(i)
        elif name2.loc[i] == 'ZnCl2':
            #pHなし
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Zn2+", d], ["Cl-",f]])
                salt2_is.append(salt_a.get_ionic_strength())

                salt2_index.append(i)
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Zn2+", d], ["Cl-",f]],pH=c)
                salt2_is.append(salt_a.get_ionic_strength())

                salt2_index.append(i)
        elif name2.loc[i] == 'KCl':
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["K+", d], ["Cl-",d]])
                salt2_is.append(salt_a.get_ionic_strength())

                salt2_index.append(i)
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Na+",d],["Cl-",d]],pH=c)
                salt2_is.append(salt_a.get_ionic_strength())

                salt2_index.append(i)
                
        elif name2.loc[i] == 'NaCl':
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]])
                salt2_is.append(salt_a.get_ionic_strength())

                salt2_index.append(i)
        
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a]],pH=c)
                salt2_is.append(salt_a.get_ionic_strength())

                salt2_index.append(i)
        
    #塩が計算できないやつ（尿素？とか）
    else:
        salt2_is.append(salt_a.get_ionic_strength())
        salt2_index.append(i)

#リスト→辞書→pandas
salt2_dict={}
for i in range(len(salt_unit2)):

    aa=str(salt2_is[i].magnitude)
    salt2_dict[salt2_index[i]]=aa
df_b=pd.DataFrame(salt2_dict,index=salt2_dict.keys())
df_salt2_is=pd.DataFrame(df_b.T.iloc[:,0])
df_salt2_is=df_salt2_is.rename(columns={1:0})
df_salt2_is.head(2)

"3種塩"
salt_conc3 = salt_conc[salt_conc[3].isna() & salt_conc[2].notna()]
salt_unit3 = salt_unit[salt_unit.index.isin(salt_conc3.index)]
salt_unit3[2] = salt_unit3[2].fillna("mM/ZnCl2") #(0704)欠損していたので、元ファイル参照に埋めた。(0708: 元ファイルの方を直したので、次多分消してもいい)
#塩の種類ごとのconc
conc3_0, conc3_1, conc3_2 = salt_conc3[0], salt_conc3[1], salt_conc3[2]
#塩の種類ごとの単位
unit3_0, unit3_1, unit3_2 = salt_unit3[0].str.split("/", expand = True)[0], salt_unit3[1].str.split("/", expand = True)[0], salt_unit3[2].str.split("/", expand = True)[0]
#塩の種類
name3_0, name3_1, name3_2 = salt_unit3[0].str.split("/", expand = True)[1], salt_unit3[1].str.split("/", expand = True)[1], salt_unit3[2].str.split("/", expand = True)[1]

"計算"
salt3_is = []
salt3_index = []
for i in salt_unit3.index:
    #2価のイオンはname3_1の方にしかないです
    a=conc3_0.loc[i]+" "+unit3_0.loc[i]
    d=conc3_2.loc[i]+" "+unit3_1.loc[i] #2価のイオンを含む
    e=(float(conc3_2.loc[i])*2) #2価なので、濃度2倍に設定
    f=str(e)+" "+unit3_1.loc[i]
    
    g = conc3_2.loc[i]+" "+unit3_2.loc[i]
    
    h2 = (float(conc3_2.loc[i])*2)
    h=str(h2)+" "+unit3_2.loc[i]
    #h: 3コメの濃度のにヴァイ

    if name3_0.loc[i]=="NaCl" :
        
        if (name3_1.loc[i] == "MgCl2") & (name3_2.loc[i] == "ZnCl2"):
            #pHなし
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f], ["Zn2+", g], ["Cl", h]]) #g: 濃度1倍, h: 濃度2倍
                salt3_is.append(salt_a.get_ionic_strength())
                salt3_index.append(i)
                print(i)
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
                #print(i)
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a],["Mg2+", d],["Cl-", f], ["Zn2+", g], ["Cl", h]], pH= c) #g: 濃度1倍, h: 濃度2倍
                salt3_is.append(salt_a.get_ionic_strength())
                salt3_index.append(i)
        if (name3_1.loc[i] == "KCl") & (name3_2.loc[i] == "MgCl2"):
            #pHなし
            if pd.isnull(df_chin.pH.loc[i]):
            #ph3.pH.loc[f2["index"].loc[i]].isna():
                salt_a=pyEQL.Solution([["Na+",a],["Cl-",a], ["K+", d], ["Cl-",d ],["Mg2+", g],["Cl-", h]]) #g: 濃度1倍, h: 濃度2倍
                salt3_is.append(salt_a.get_ionic_strength())
                salt3_index.append(i)
                print(i)
            #pHあり
            else:
                c=math.floor(df_chin.pH.loc[i]) 
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

salt3_dict={}
for i in range(len(salt3_is)):

    aa=str(salt3_is[i].magnitude)
    salt3_dict[salt3_index[i]]=aa

df_b=pd.DataFrame(salt3_dict,index=salt3_dict.keys())
df_salt3_is=pd.DataFrame(df_b.T.iloc[:,0])
df_salt3_is=df_salt3_is.rename(columns={df_salt3_is.columns[0]:0})

#塩4種類のやつにphosphateあり→pyEQLで計算できないのでパス→np.nan
#塩入れてない系のイオン強度は0で良いのかどうか→とりあえずいいか

"計算したイオン強度を大元のデータフレームにカラム追加する"
df_salt_is = pd.concat([df_salt1_is, df_salt2_is, df_salt3_is], axis = "index")
df_chin["ionic_strength"] = df_salt_is

df_chin.ionic_strength = df_chin.ionic_strength.fillna(0)
df_chin.ionic_strength[df_chin.salt_unit_name.str.contains("phosphate", na = False)] = np.nan

df_chin.to_csv("./preprocessing_results/rnap_preprocessing_is.csv", index = False)
