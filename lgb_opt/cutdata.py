import argparse
import numpy as np
import pandas as pd
import yaml
import os


def main():
    with open("./config_.yaml",'r')as f:
        args = yaml.safe_load(f)
    target_label = args["target_label"]
    group_label = args["group_label"]
    random_state = args["random_state"]
    data_dir = args["data_dir"]
    file_name = args["file_name"]
    mor_class = np.array(args["mor_class"])
    max_num = args["max_num"]
    min_num = args["min_num"]    


    "下限設定なしでmax_num以下のデータ選択"
    def filtered_data(df, select_len = max_num):
    #def filtered_data(df = df_chin, select_len =  (df_prepro[df_prepro.group_label == group].shape[0] * 2)):
        selected_list = []
        for group in df.group_label.unique():
            if df[df.group_label == group].shape[0] >= select_len:
                selected_list.append(df[df.group_label == group].sample(select_len, random_state = 42,))
            else:
                selected_list.append(df[df.group_label == group])
        df_sel = pd.DataFrame()
        for select in selected_list:
            df_sel = pd.concat([df_sel, select], axis = "index")

        return df_sel

    "上限下限設定、データ選択"
    def selected_data(min_num = min_num, max_num = max_num):
        df_chin_1 = df_chin_0[df_chin_0.mor_label == 0]
        index_1 = df_chin_1.group_label.value_counts()[df_chin_1.group_label.value_counts()>=min_num].index.values
        df_chin_ = df_chin_0[df_chin_0.mor_label == 1]
        index_0 = df_chin_.group_label.value_counts()[df_chin_.group_label.value_counts()>=min_num].index.values
        index_sel =  np.intersect1d(index_0,index_1)
        
        df_chin_in_0 = df_chin_[df_chin_.group_label.isin(index_sel)]
        df_chin_in_1 = df_chin_1[df_chin_1.group_label.isin(index_sel)]
        
        df_in_1 = filtered_data(df = df_chin_in_0, select_len = max_num)
        df_in_0 = filtered_data(df = df_chin_in_1, select_len = max_num)
        df_in_10 = pd.concat([df_in_1, df_in_0], axis = "index").reset_index(drop = True)
        return df_in_10

  


    mor = [["solute", "liquid", "solid"][i] for i in mor_class]
    # for idx, name in enumerate(df_name):
    files = data_dir + file_name + '.csv'
    df_chin = pd.read_csv(files, index_col=False) 
    df_chin_0 = df_chin[df_chin[target_label].isin(mor_class)] #予測する形態指定(solute, liquid, solid)
    "上限設定"


    df_chin_sel = filtered_data(df_chin_0, select_len = max_num)
    
    print(df_chin_sel.group_label.value_counts())
    print("論文数: ", len(df_chin_sel.group_label.unique()))
    print("min_num:", min_num, ", max_num: ", max_num)
    print(df_chin_sel.shape)
    print(df_chin_sel.mor_label.value_counts())


    files_path = data_dir + file_name 
    os.makedirs(files_path, exist_ok = True)
    df_chin_sel.to_csv(f"{files_path}/sel_{file_name}_{min_num}_{max_num}.csv", index = False)
    print(f"{files_path}/sel_{file_name}_{min_num}_{max_num}.csv")

if __name__ == "__main__":
    main()