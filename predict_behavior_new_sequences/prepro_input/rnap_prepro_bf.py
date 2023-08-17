def seq2features(seq_dict):
    import numpy as np
    import pandas as pd
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    weight_list={}
    aa_count_list={}
    aa_count_per_list={}
    aromaticity_list={}
    gravy_list={}
    instability_list={}
    flexibility_list={}
    isopt_list={}
    sec_structure_list={}
    for i in range(seq_dict.shape[0]):
        if seq_dict[i]!=None:
            annalysed_seq=ProteinAnalysis(seq_dict[i])

            weight_list[i]=annalysed_seq.molecular_weight()
            aa_count_list[i]=annalysed_seq.count_amino_acids()
            aa_count_per_list[i]=annalysed_seq.get_amino_acids_percent()
            aromaticity_list[i]=annalysed_seq.aromaticity()
            gravy_list[i]=annalysed_seq.gravy()
            instability_list[i]=annalysed_seq.instability_index()
            flexibility_list[i]=annalysed_seq.flexibility()
            isopt_list[i]=annalysed_seq.isoelectric_point()
            sec_structure_list[i]=annalysed_seq.secondary_structure_fraction()
        else:
            assert seq_dict[i]!=None, "seq contains nan"
            continue
    #list to dataframe
    d2={}
    for k,v in weight_list.items():   # 一度pd.Seriesに変換
        d2[k]=pd.Series(v)
    d2=pd.DataFrame(d2)
    d2=d2.T
    d2=d2.rename(columns={0:"weight"})

    d3={}
    for k,v in aa_count_per_list.items():   # 一度pd.Seriesに変換
        d3[k]=pd.Series(v)
    d3=pd.DataFrame(d3)
    d3=d3.T

    d4={}
    for k,v in aromaticity_list.items():   # 一度pd.Seriesに変換
        d4[k]=pd.Series(v)
    d4=pd.DataFrame(d4)
    d4=d4.T
    d4=d4.rename(columns={0:"aromaticity"})

    d5={}
    for k,v in instability_list.items():   # 一度pd.Seriesに変換
        d5[k]=pd.Series(v)
    d5=pd.DataFrame(d5)
    d5=d5.T
    d5=d5.rename(columns={0:"instability"})

    d6={}
    for k,v in gravy_list.items():   # 一度pd.Seriesに変換
        d6[k]=pd.Series(v)
    d6=pd.DataFrame(d6)
    d6=d6.T
    d6=d6.rename(columns={0:"gravy"})

    d7={}
    for k,v in flexibility_list.items():   # 一度pd.Seriesに変換
        d7[k]=pd.Series(v)
    d7=pd.DataFrame(d7)
    d7=d7.T
    d7=d7.rename(columns={0:"flexibility"})
    d7["average"]=d7.mean(axis="columns")

    d8={}
    for k,v in isopt_list.items():   # 一度pd.Seriesに変換
        d8[k]=pd.Series(v)
    d8=pd.DataFrame(d8)
    d8=d8.T
    d8=d8.rename(columns={0:"isopt"})
    d9={}
    for k,v in sec_structure_list.items():   # 一度pd.Seriesに変換
        d9[k]=pd.Series(v)
    d9=pd.DataFrame(d9)
    d9=d9.T
    d9=d9.rename(columns={0:"helix", 1:"turn", 2:"sheet"})
    #concat all features
    df_aa_analysis1=pd.concat([d2,d3,d4,d5,d6,d7["average"],d8,d9],axis="columns")
    
    return df_aa_analysis1