#!/bin/bash
python ./prepro_ex_protein_feature.py
bash ./run_mf.sh
python ./prepro_rna_feature.py
python ./prepro_input_file.py
