#!/bin/bash
python ~/MathFeature/methods/ExtractionTechniques.py -i ./prepro_results/rna_seq.fasta -o ./mf_result/mf_3mer.csv -t kmer -seq 2 < mf_input.txt
python ~/MathFeature/methods/FickettScore.py -i ./prepro_results/rna_seq.fasta -o ./mf_result/mf_fickett.csv -seq 2
python ~/MathFeature/methods/CodingClass.py -i ./prepro_results/rna_seq.fasta -o ./mf_result/mf_orf.csv 
python ~/MathFeature/methods/TsallisEntropy.py -i ./prepro_results/rna_seq.fasta -o ./mf_result/mf_tsallis_entropy.csv -k 2 -q 0.1
python ~/MathFeature/methods/EntropyClass.py -i ./prepro_results/rna_seq.fasta -o ./mf_result/mf_shannon.csv -k2 -e Shannon
python ~/MathFeature/methods/FourierClass.py -i ./prepro_results/rna_seq.fasta -o ./mf_result/mf_fourier_binary.csv -r 1
python ~/MathFeature/methods/FourierClass.py -i ./prepro_results/rna_seq.fasta -o ./mf_result/mf_fourier_zcurve.csv -r 2
python ~/MathFeature/methods/FourierClass.py -i ./prepro_results/rna_seq.fasta -o ./mf_result/mf_fourier_real.csv -r 3



