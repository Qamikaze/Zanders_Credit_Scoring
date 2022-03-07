# Zanders_Credit_Scoring
This repository contains all the code for the models and data preprocessing used by Team Zanders-B. It consists of the following sections:

- **Data Prepocessing**: this section contains all the files used for prepocessing the data.
  - **GetFeatures.py**: This file performs a variable selection by removing highly correlated features.
  - **KNN_imputer.py**: This file imputes missing values by using a KNN-estimator.

- **Interpret**: this section contains the file which uses model-agnostic methods to interpret the data.
  - **ALE_LIME.py**: This file implements ALE for global model-agnostic interpretation and LIME for local model-agnostic explanation.

- **McNemar**: this section contains the file which uses McNemar's tests to compare the output among models.
  - **McNemar.py**: This file is used to perform the McNemar's test to compare the performance of different classification models.

- **Models**: this section contains all the used models.
  - **AugBoost_RF**: this section contains the files for the AugBoost-RF model.
    - **AugBoostRandomForest.py**: This code performs the AugBoostRF algorithm. It is tweeked from the original AugBoostNN function, see https://github.com/augboost-anon/augboost.
    - **AugmentationUtils.py**: This code augments the features for the AugBoostRF. Therfore, it is imported by AugBoostRandomForest.py.  It is an adjusted version of the AugmentationUtils on https://github.com/augboost-anon/augboost/blob/master/AugmentationUtils.py.
    - **Run_augboost.py**: This code provides output for the AugBoost-RF model. It imports the function AugBoostClassifier from AugBoostRandomForest.py.
  - **XGBoost_RF**: this section contains the files for the new XGBoost-RF model.
    - **AugmentationUtils.py**: This code augments the features for the XGBoost-RF algorithm using a Random Forest.  It is an adjusted version of the AugmentationUtils on https://github.com/augboost-anon/augboost/blob/master/AugmentationUtils.py.
    - **RUN_XGBOOST_LAST.py**: The code used to run the XGBoost-RF algorithm using the XGBOOST_LAST.py code and get the performance measures of the model.
    - **XGBOOST_LAST.py**: This code performs the XGBoost-RF algorithm. It has as a basis the original AugBoostNN function, see https://github.com/augboost-anon/augboost and is adjusted accordingly.
  - **GBDT.py**: This code provides output for the GBDT algorithm.
  - **LightGBM.py**: This code provides the output of the LightGBM algorithm.
  - **LogisticRegression.py**: This code is used to perform the Weight of Evidence on the data and subsequently perform the logistic regression and get its output.
  - **RandomForest**: This file tunes a random forest classifier using random grid search and saves the model with the best hyperparameters.
  - **XGBoost**: This file tunes an XGBoost classifier using random grid search and saves the model with the best hyperparameters.
 
