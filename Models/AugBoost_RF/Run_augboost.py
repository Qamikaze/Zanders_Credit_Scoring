import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from hmeasure import h_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from AugBoostNN import AugBoostClassifier as ABC


#import matplotlib.pyplot as plt
#import shap

# =============================================================================
# Performance metrics
# =============================================================================
start_time = datetime.now()

def partial_gini(y, p):
    
    # Select probabilites < 0.4
    output = pd.DataFrame()
    output['y']=y
    output['p']=p
    under4 = output[output["p"] < 0.4]
    y = under4['y']
    p = under4['p']
    
    # Calculate Gini index
    # see https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation
    assert (len(y) == len(p))
    all = np.asarray(np.c_[y, p, np.arange(len(y))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(y) + 1) / 2.
    
    return giniSum / len(y)

def get_performance(y, y_f, p):
    
    # print performance metrics
    print('PG is ', partial_gini(y, p))
    print('BS is ', brier_score_loss(y, p))
    print('AUC is ', roc_auc_score(y, p))
    print('H-measure is ', h_score(y.values, p))
    print(confusion_matrix(y, y_f))
    
# =============================================================================
# Data splitting
# =============================================================================

def get_data(small):
    train = pd.read_csv('C:/Users/svenh/OneDrive/Documents/Master/Seminar Case Studies/train_data.csv.gz', compression='gzip',  header=0, sep=',', quotechar='"')
    test = pd.read_csv('C:/Users/svenh/OneDrive/Documents/Master/Seminar Case Studies/test_data.csv.gz', compression='gzip',  header=0, sep=',', quotechar='"')
    
    if small == True:
        n = 10000
        train = train.head(n=n)
        test = test.head(n=n)
        
    # Factorize data
    train['country_code'] = pd.factorize(train['country_code'])[0]
    train['industry_code'] = pd.factorize(train['industry_code'])[0]
    train['size_class'] = pd.factorize(train['size_class'])[0]
    train['status_year'] = pd.factorize(train['status_year'])[0]
    
    test['country_code'] = pd.factorize(test['country_code'])[0]
    test['industry_code'] = pd.factorize(test['industry_code'])[0]
    test['size_class'] = pd.factorize(test['size_class'])[0]
    test['status_year'] = pd.factorize(test['status_year'])[0]
    
    # Seperate labels and covariates
    X = train.drop(['y'], axis=1)
    y = train['y']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=15/85, random_state=42)
    
    X_train = pd.concat([X_train,X_val], ignore_index=True)
    y_train = pd.concat([y_train,y_val], ignore_index=True)  
    
    X_test = test.drop(['y'], axis=1)
    y_test = test['y']
    
    c = ['country_code',
         'industry_code',
         'size_class',
         'status_year',
         'intangible_fixed_assets_2',
         'stock_2',
         'loans_0',
         'creditors_2',
         'costs_of_goods_sold_2',
         'pl_for_period_net_income_0',
         'interest_paid_0',
         'interest_paid_1',
         'interest_paid_2',
         'ebitda_0',
         'ebitda_2',
         'total_debt_0',
         'total_debt_2',
         'tangible_net_worth_2',
         'turnover_growth_0',
         'gross_margin_0',
         'gross_margin_1',
         'gross_margin_2',
         'operating_margin_0',
         'operating_margin_1',
         'operating_margin_2',
         'return_on_sales_0',
         'return_on_sales_1',
         'return_on_sales_2',
         'return_on_capital_employed_0',
         'return_on_capital_employed_1',
         'quick_ratio_0',
         'quick_ratio_1',
         'quick_ratio_2',
         'stock_days_0',
         'stock_days_1',
         'debtor_days_0',
         'debtor_days_1',
         'creditor_days_0',
         'creditor_days_1',
         'gearing_0',
         'gearing_1',
         'gearing_2',
         'solvency_0',
         'solvency_1',
         'solvency_2',
         'debt_ebitda_ratio_0',
         'debt_ebitda_ratio_1',
         'debt_ebitda_ratio_2',
         'interest_coverage_ratio_0',
         'interest_coverage_ratio_1',
         'interest_coverage_ratio_2']
    
    
    return X_train[c], y_train, X_test[c], y_test

X_train, y_train, X_test, y_test = get_data(False)
#%%
# Use balanced weights
weight = y_train*len(y_test)/sum(y_test)/2
weight[np.where(weight == 0)[0]] = 1

print('Data loading duration: {}'.format(datetime.now() - start_time))

start_time = datetime.now()

num_rows, num_cols = X_train.shape

from AugBoostRandomForest import AugBoostClassifier as ABC

model = ABC(n_estimators=11, max_epochs=100, learning_rate=0.1, \
    n_features_per_subset=3, trees_between_feature_update=10,\
    augmentation_method='rf', save_mid_experiment_accuracy_results=False,loss='exponential')

model.fit(X_train, y_train, sample_weight=weight)
import pickle
#Save model to file
pkl_filename = "GDBT_RF.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# Load model from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
Get test performance
pred_values = model.predict(X_test)
prob_values = model.predict_proba(X_test)[:, 1]
get_performance(y_test, pred_values, prob_values)
prob_values.to_csv('prob_values.csv')
print('Model Duration: {}'.format(datetime.now() - start_time))


# check performance with DT
from sklearn.tree import DecisionTreeRegressor
p_values = model.predict_proba(X_train)[:, 1]
reg = DecisionTreeRegressor().fit(X_train, p_values)
y_pred = reg.predict(np.array(X_train))

from sklearn.metrics import r2_score
r = r2_score(p_values, y_pred)
print(r)
reg.get_depth()

# get importance
importance = reg.feature_importances_

ordered_importances = pd.DataFrame()
ordered_importances['feature'] = X_train.columns
ordered_importances['importance'] = importance 
ordered_importances = ordered_importances.sort_values(ascending=False, by=['importance'])

n =10
some = ordered_importances.head(n)
sum(ordered_importances['importance'].head(n))



