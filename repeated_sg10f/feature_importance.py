#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
# データセットの特徴量数を設定する
def main():
    num_features = 131
    # 平均特徴量重要度を格納するリストを初期化
    avg_fi_list = []
    model_name = "AdaBoostClassifier"
    # 10-Foldをループで回す
    for kf_state in range(10, 101, 10):
        avg_feature_importance = np.zeros(num_features)
        model_path = f"result_repeated_StratifiedGroupKFold/{model_name}/kf_state_{kf_state}/model_list_ml_rnapsec_aa.pickle"
        # モデルを読み込む
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        for fold_num in range(0, 10):
            model = loaded_model[fold_num]
            # 特徴量重要度を取得する
            fold_feature_importance = model.feature_importances_
            # 平均特徴量重要度に加算する
            avg_feature_importance += fold_feature_importance

        # 平均特徴量重要度を計算する
        avg_feature_importance /= 10

        avg_fi_list.append(avg_feature_importance)
    cols = pd.read_csv("../data/ml_rnapsec_aa.csv", index_col=False).drop(["mor_label", "aa_label", "rnapsec_all_col_idx",
        'protein_sequence', 'rna_sequence', 'protein_name', 
        'aa_rna_label'], axis = "columns").columns
    cols = cols.str.replace("rna_rna_", "(RNA) ")
    cols = cols.str.replace("protein_conc_log", "Protein conc.")
    cols = cols.str.replace("rna_conc_log", "RNA conc.")
    cols = cols.str.replace("protein_", "(Protein) ")
    cols = cols.str.replace("ionic_strength", "Ionic strength")
    cols = cols.str.replace("ph", "pH")
    cols = cols.str.replace("temperature", "Temperature")
    feature_importance = pd.DataFrame(avg_fi_list, columns=cols).mean(axis="index").sort_values(ascending=False)
    top_10_features = feature_importance.nlargest(10)[::-1]

    df_result_top10 = pd.DataFrame(avg_fi_list, columns=cols)
    df_result_top10 = df_result_top10.loc[:, df_result_top10.columns.isin(top_10_features.index)]

    err = np.std(df_result_top10)
    # 横棒グラフを作成
    plt.figure(figsize=(7, 5), dpi=300) 
    plt.barh(top_10_features.index, top_10_features.values,xerr = err, color='b')
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title( "Top 10 Average Feature Importances of the AdaBoost model", fontsize=14, pad=8)
    # 軸のラベルのフォントサイズを調整
    plt.xticks(fontsize=10)
    plt.xlim(0)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"result_repeated_StratifiedGroupKFold/{model_name}/avg_feature_importance.png")
    plt.close()
if __name__=="__main__":
    main()