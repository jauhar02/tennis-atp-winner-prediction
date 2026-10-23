# ATP Tennis Match Winner Prediction

Code and derived features for the study "A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes".

## Data and attribution
Match data are from Jeff Sackmann's ATP dataset (https://github.com/JeffSackmann), licensed CC BY-NC-SA 4.0.
This study used the archived copy at https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive.
The derived features and code in this repository are released under the same license (CC BY-NC-SA 4.0), with attribution to Jeff Sackmann. Non-commercial use only.

## Contents
- prediksi_tenis_atp.ipynb: full pipeline (cleaning, features, tuning, evaluation, SHAP, Gradio app)
- model_tenis/: derived features, results tables, and saved models

## Environment
Google Colab, Python 3.13.16, scikit-learn 1.6.1, xgboost 3.4.1, shap 0.52.0, pandas 2.2.3, numpy 2.1.3, statsmodels 0.15.0. Random seed: 42.
