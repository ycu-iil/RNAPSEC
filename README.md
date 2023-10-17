# About this repository
This repository provides RNAPSEC and prediction models mentioned in [bioRxiv]( https://doi.org/10.1101/2023.06.01.543215). RNAPSEC is a dataset constructed by re-collecting data from RNAPhaSep (Zhu et al., 2021), with RNA and protein-related LLPS experiments as entries. The first prediction model predicts whether a given protein and RNA will undergo LLPS under specified conditions, based on the protein sequence, RNA sequence and experimental conditions. The second prediction model predicts the class of experimental conditions under which a protein and RNA will undergo LLPS from the protein sequence and RNA sequence. 
**Note:** Jupyter notebook for the prediction model that predict LLPS behavior is available at [Google Colaboratory](https://colab.research.google.com/drive/13n6yXMnmtuKbZ6imWzPfv4M_k3ZxVgHI#scrollTo=qoSvAlcNoqEn)

# About RNAPSEC
- RNAPSEC is a dataset with experiments as entries, which was constructed by re-collecting information on LLPS experiments from the original papers of the data contained in RNAPhaSep. The protein concentration, RNA concentration, pH, salt concentration and temperature used in LLPS experiments with one type of protein and RNA and its experimental results were manually re-collected. Experimental information other than the above is included directly from RNAPhaSep.
- The original dataset is available at "/data/rnapsec.xlsx" 
- The preprocessed dataset is available at ``` /predict_behavior_new_sequences/```
- Detailed information about RNAPSEC was described at ``` /data/README.md```
# Confirmed
Python: 3.11.0

# Setup
Create new environment and move into it.
``` 
conda create -n rnapsec python==3.11
source activate rnapsec
``` 
Clone github directory and install requiremental libraries. 
``` 
git clone git@github.com:ycu-iil/RNAPSEC.git
git clone https://github.com/Bonidia/MathFeature.git 
cd RNAPSEC
pip install -r requirement.txt
``` 
-----
# Usage by example
## Predicting LLPS behavior using pretrained model
- By running the following codes, you can predict whether the specified protein and RNA in ``` /predict_behavior_new_sequences/prepro_input/example.xlsx``` can undergo LLPS under the specified experimental conditions. 
    - The example file contains the protein sequence (single letter sequence), RNA sequence (single letter sequence), and experimental conditions (protein concentration, RNA concentration, temperature, pH, ionic strength) in a regular format. 
    - You can set different protein or RNA sequences and experimental conditions by updating the context in the example.xlsx.
- The prediction result, LLPS probability, and a phase diagram that shows the predicted outcomes within a log scale range of ±0.5 for the input protein concentration and RNA concentration will be outputed to ``` /predict_behavior_new_sequences/```.

### Running: 
1. Prepare the input file 
    
    ``` 
    cd ./predict_behavior_new_sequences
    bash preprocessing.sh 
    ``` 
2. Predict input file and constructing phase diagram using pretrained model
    ``` 
    python prediction.py
    ``` 
## Predicting experimental conditions using pretrained model
- Running the following code will give the experimental conditions under which the protein and RNA specified in the ``` /predict_condition_new_sequences/prepro_input/example.xlsx``` will undergo LLPS.
    - You can set different protein or RNA sequences and experimental conditions by updating the context in the example.xlsx.
- Results will be outputed to``` /predict_condition_new_sequences/```

### Running:
1.  Prepare a input file
    ``` 
    cd ./predict_condition_new_sequences/
    bash preprocessing.sh
    ``` 
2.  Predict the preprocessed input file
    ``` 
    python prediction.py
    ``` 
# Evaluation of the model performances through cross-validation
##  Preprocessing of RNAPSEC
1. Preprocessig of RNAPSEC for model developments
    ``` 
    cd preprocessing
    bash preprocesssing_rnapsec.sh
    cd ../
    ``` 
##  Evaluation of the model that predict LLPS behavior
1. Training and evaluation through Leave One Group Out cross-validation
    ``` 
    python logocv.py
    python phase_diagram_logocv.py #phase diagrams
    python feature_importances_logocv.py #feature importances
    ``` 
2. Training and evaluation through Repeated Group 10-Fold cross-validaton
    ``` 
    cd repeated_sgkf
    python repeated_cv.py
    python split_data_sgkf.py phase_diagram.py
    python feature_importance.py
    ``` 
## Evaluation of the model that predict LLPS conditions
1. Training and evaluation through Group 10-Fold cross-validation 
    ``` 
    cd ./predict_conditions_cross_validation
    python chain.py
    ``` 
# Publication
Chin, K., Ishida, S., & Terayama, K. (2023). Predicting condensate formation of protein and RNA under various environmental conditions. bioRxiv (Cold Spring Harbor Laboratory). https://doi.org/10.1101/2023.06.01.543215

# List of files
- data: RNAPSEC before and after preprocessing #originalとpreprocessing 
- preprocessing_rnapsec: preprocessing files (scripts and related-files) for RNAPSEC
- predict_behavior_new_sequence: pre-trained model and running scripts to predict LLPS behavior
- predict_condition_new_sequence: pre-trained model and running scripts to predict experimental conditions for LLPS
- predict_behavior_cross_validation, repeated_sgkf, predict_condition_cross_validation: scripts used in model evaluation
- requirements.txt: Dependencies
- README.md: Documentation



