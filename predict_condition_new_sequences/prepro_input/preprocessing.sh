#!/bin/bash
python ./prepro_input/prepro_ex_protein_feature.py
bash ./prepro_input/run_mf.sh
python ./prepro_input/prepro_rna_feature.py
python ./prepro_input/prepro_input_file.py
