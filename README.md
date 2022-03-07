# Zanders_Credit_Scoring
This repository contains all the code for the models and data preprocessing used by Team Zanders-B. It consists of the following sections:

- Data Prepocessing: this section contains all the files used for prepocessing the data.

- Interpret: this section contains the file which uses model-agnostic methods to interpret the data.

- McNemar: this section contains the file which uses McNemar's tests to compare the output among models.
 This file is used to perform the McNemar's test to compare the performance of different classification models.

- Models: this section contains all the used models.

- XGBoost-RF: The code for the new XGBoost-RF model.
  - AugmentationUtils.py: This code augments the features for the XGBoost-RF algorithm using a Random Forest.
  - RUN_XGBOOST_LAST.py: The code used to run the XGBoost-RF algorithm using the XGBOOST_LAST.py code and get the performance measures of the model.
  - XGBOOST_LAST.py: This code performs the XGBoost-RF algorithm. It has as a basis the original AugBoostNN function, see https://github.com/augboost-anon/augboost and is adjusted accordingly.
 
