#rnap_prepro_1 で作成したrna配列を含んだfastaファイルを入力に使用
#各計算結果が個々のcsvファイルに出力
#5_rnap_concat_mf_resultで統合

#python /home/chin/download/MathFeature/preprocessing/preprocessing.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/prepro.fasta
#kmer, orf, fickett
python /home/chin/download/MathFeature/methods/ExtractionTechniques.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/mf_3mer.csv -t kmer -seq 2
3
python /home/chin/download/MathFeature/methods/FickettScore.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/mf_fickett.csv -seq 2
python /home/chin/download/MathFeature/methods/CodingClass.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/mf_orf.csv 
#entropy
python /home/chin/download/MathFeature/methods/TsallisEntropy.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/mf_tsallis_entropy.csv -k 2 -q 0.1
python /home/chin/download/MathFeature/methods/EntropyClass.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/mf_shannon.csv -k2 -e Shannon

#fourier
python /home/chin/download/MathFeature/methods/FourierClass.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/mf_fourier_binary.csv -r 1
python /home/chin/download/MathFeature/methods/FourierClass.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/mf_fourier_zcurve.csv -r 2
python /home/chin/download/MathFeature/methods/FourierClass.py -i ./prepro_results/rnap_rnaseq.fasta -o ./mf_result/mf_fourier_real.csv -r 3
