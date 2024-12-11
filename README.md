# Aihack-Thailand-2023

A machine learning-based *credit risk model* designed to predict **Long-Overdue Debtors (LOD)**.

**Awards**: Winner Prize and Best Performance (AUC) Prize.

> Note: Sensitive information is deleted once the competition is over to comply with licensing requirements.

## Directory Structure
```txt
.
├── LICENSE
├── README.md
├── gitignore.txt
├── models
│   └── catboost_info
│       ├── catboost_training.json
│       ├── learn
│       │   └── events.out.tfevents
│       ├── learn_error.tsv
│       ├── time_left.tsv
│       └── tmp
│           ├── cat_feature_index.227883a1-ea0c9bc2-8996f9c9-6c47af9e.tmp
│           ├── cat_feature_index.810dd4f5-95a97112-e3bd4a75-13b41f3f.tmp
│           ├── cat_feature_index.8b56a70d-82f739a8-2c403b40-6eef2ce1.tmp
│           ├── cat_feature_index.9c35c1aa-3307d4d9-753fc6c8-49537f4a.tmp
│           ├── cat_feature_index.af190e2c-1d7e3e3d-2a8530f5-9dad04b4.tmp
│           ├── cat_feature_index.c25e503b-ef9397df-e7ec44aa-35212493.tmp
│           ├── cat_feature_index.d32b9db0-bce4fad5-64aa9c2a-5cdc4627.tmp
│           ├── cat_feature_index.edcc8c88-cd1beb86-9682a877-a65ff60d.tmp
│           └── cat_feature_index.fc691468-5bfa1911-327d591-628b14a5.tmp
├── notebooks
│   ├── 1stRound
│   │   ├── LOD.ipynb
│   │   ├── LOD_231214.ipynb
│   │   ├── LOD_231215-0300.ipynb
│   │   ├── LOD_231215-1030.ipynb
│   │   ├── LOD_pycaret.ipynb
│   │   ├── LOD_regression.ipynb
│   │   ├── LOD_regression2.ipynb
│   │   ├── Tee_231214-1700.ipynb
│   │   └── Tee_231214-2200.ipynb
│   ├── 2ndRound
│   │   ├── LOD_231215-1300.ipynb
│   │   └── Load.ipynb
│   ├── Titanic
│   │   ├── titanic-Pre_Practice.ipynb
│   │   ├── titanic-task1.ipynb
│   │   └── titanic-task2.ipynb
│   ├── main
│   │   ├── Hackathon_AI_Crack_Crack.ipynb
│   │   └── Hackathon_AI_Crack_Crack_Official_Sent.ipynb
│   └── unclassified_version_1.ipynb
├── results
│   ├── 1stRound
│   │   ├── output_10min.csv
│   │   ├── output_10min_WeightedEnsemble_L2.csv
│   │   ├── output_120min_CatBoost_BAG_L1.csv
│   │   ├── output_120min_CatBoost_BAG_L2.csv
│   │   ├── output_120min_CatBoost_r177_BAG_L1.csv
│   │   ├── output_120min_LightGBMXT_BAG_L2.csv
│   │   ├── output_120min_LightGBM_BAG_L2.csv
│   │   ├── output_120min_WeightedEnsemble_L2.csv
│   │   ├── output_120min_WeightedEnsemble_L3.csv
│   │   ├── output_15min_CatBoost_BAG_L2.csv
│   │   ├── output_15min_WeightedEnsemble_L3.csv
│   │   ├── output_15minlast_CatBoost_BAG_L1.csv
│   │   ├── output_15minlast_CatBoost_BAG_L2.csv
│   │   ├── output_15minlast_CatBoost_r177_BAG_L1.csv
│   │   ├── output_15minlast_CatBoost_r177_BAG_L2.csv
│   │   ├── output_15minlast_CatBoost_r9_BAG_L1.csv
│   │   ├── output_15minlast_CatBoost_r9_BAG_L2.csv
│   │   ├── output_15minlast_LightGBMLarge_BAG_L1.csv
│   │   ├── output_15minlast_LightGBMLarge_BAG_L2.csv
│   │   ├── output_15minlast_LightGBMXT_BAG_L1.csv
│   │   ├── output_15minlast_LightGBMXT_BAG_L2.csv
│   │   ├── output_15minlast_LightGBM_BAG_L1.csv
│   │   ├── output_15minlast_LightGBM_BAG_L2.csv
│   │   ├── output_15minlast_LightGBM_r131_BAG_L1.csv
│   │   ├── output_15minlast_LightGBM_r131_BAG_L2.csv
│   │   ├── output_15minlast_LightGBM_r96_BAG_L1.csv
│   │   ├── output_15minlast_LightGBM_r96_BAG_L2.csv
│   │   ├── output_15minlast_WeightedEnsemble_L2.csv
│   │   ├── output_15minlast_WeightedEnsemble_L3.csv
│   │   ├── output_15minlast_XGBoost_BAG_L1.csv
│   │   ├── output_15minlast_XGBoost_BAG_L2.csv
│   │   ├── output_15minlast_XGBoost_r33_BAG_L1.csv
│   │   ├── output_15minlast_XGBoost_r33_BAG_L2.csv
│   │   ├── output_1h.csv
│   │   ├── output_20min_CatBoost_BAG_L1.csv
│   │   ├── output_20min_ExtraTreesEntr_BAG_L1.csv
│   │   ├── output_20min_ExtraTreesGini_BAG_L1.csv
│   │   ├── output_20min_KNeighborsDist_BAG_L1.csv
│   │   ├── output_20min_KNeighborsUnif_BAG_L1.csv
│   │   ├── output_20min_LightGBMLarge_BAG_L1.csv
│   │   ├── output_20min_LightGBMXT_BAG_L1.csv
│   │   ├── output_20min_LightGBM_BAG_L1.csv
│   │   ├── output_20min_NeuralNetFastAI_BAG_L1.csv
│   │   ├── output_20min_NeuralNetTorch_BAG_L1.csv
│   │   ├── output_20min_RandomForestEntr_BAG_L1.csv
│   │   ├── output_20min_RandomForestGini_BAG_L1.csv
│   │   ├── output_20min_WeightedEnsemble_L2.csv
│   │   ├── output_20min_XGBoost_BAG_L1.csv
│   │   ├── output_30min.csv
│   │   ├── output_3h.csv
│   │   ├── output_45min_CatBoost_BAG_L1-2023_12_15_0230.csv
│   │   ├── output_45min_CatBoost_BAG_L1.csv
│   │   ├── output_45min_CatBoost_BAG_L2-2023_12_15_0230.csv
│   │   ├── output_45min_CatBoost_r177_BAG_L1-2023_12_15_0230.csv
│   │   ├── output_45min_LightGBMLarge_BAG_L2-2023_12_15_0230.csv
│   │   ├── output_45min_LightGBMXT_BAG_L1-2023_12_15_0230.csv
│   │   ├── output_45min_LightGBMXT_BAG_L2-2023_12_15_0230.csv
│   │   ├── output_45min_LightGBM_BAG_L1-2023_12_15_0230.csv
│   │   ├── output_45min_LightGBM_BAG_L2-2023_12_15_0230.csv
│   │   ├── output_45min_WeightedEnsemble_L2-2023_12_15_0230.csv
│   │   ├── output_45min_WeightedEnsemble_L3-2023_12_15_0230.csv
│   │   ├── output_45min_WeightedEnsemble_L3.csv
│   │   ├── output_45min_XGBoost_BAG_L1-2023_12_15_0230.csv
│   │   ├── output_45min_XGBoost_BAG_L2-2023_12_15_0230.csv
│   │   ├── output_60min_CatBoost_BAG_L2.csv
│   │   ├── output_60min_CatBoost_r177_BAG_L2.csv
│   │   ├── output_60min_WeightedEnsemble_L2.csv
│   │   ├── output_60min_WeightedEnsemble_L3.csv
│   │   ├── output_pre1h.csv
│   │   ├── output_pyc1.csv
│   │   ├── output_pyc1_feature.png
│   │   ├── output_pyc1_feature_all.png
│   │   ├── output_pyc2.csv
│   │   ├── output_pyc3.csv
│   │   └── top
│   │       ├── d4.csv
│   │       ├── d5.csv
│   │       ├── d6.csv
│   │       ├── submission6_CatBoost_BAG_L1.csv
│   │       ├── submission6_WeightedEnsemble_L3.csv
│   │       └── submission8_WeightedEnsemble_L2.csv
│   ├── 2ndRound
│   │   ├── submission-1.csv
│   │   └── submission-2.csv
│   ├── Titanic
│   │   ├── titanic-predictions_randomforest.csv
│   │   ├── titanic-sample-submission.csv
│   │   └── titanic-submission-2023_11_29.csv
│   ├── logs.log
│   └── notes.txt
├── scripts
│   └── Titanic
│       ├── titanic-task1.py
│       └── titanic-task2.py
└── tree.txt

23 directories, 137 files
```
