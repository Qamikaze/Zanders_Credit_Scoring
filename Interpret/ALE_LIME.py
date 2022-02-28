import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from hmeasure import h_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from AugBoostRF import AugBoostClassifier as ABC
import matplotlib.pyplot as plt
# import shap
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pickle
from PyALE import ale
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
    train = pd.read_csv('train_data.csv.gz', compression='gzip',  header=0, sep=',', quotechar='"')
    test = pd.read_csv('test_data.csv.gz', compression='gzip',  header=0, sep=',', quotechar='"')
    
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

def get_data2(small):
    train = pd.read_csv('train_data.csv.gz', compression='gzip',  header=0, sep=',', quotechar='"')
    test = pd.read_csv('test_data.csv.gz', compression='gzip',  header=0, sep=',', quotechar='"')
    
    if small == True:
        n = 10000
        train = train.head(n=n)
        test = test.head(n=n)
        
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

X_train2, y_train2, X_test2, y_test2 = get_data2(False)


# =============================================================================
# Model interpreting
# =============================================================================

# load model from file
pkl_filename = "TUNED_rf.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)
    
# # check performance with DT
# p_values = model.predict_proba(X_train)[:, 1]
# reg = DecisionTreeRegressor().fit(X_train, p_values)
# y_pred = reg.predict(np.array(X_train))
# r = r2_score(p_values, y_pred)
# print(r)
# reg.get_depth()

# # get importance
# importance = reg.feature_importances_
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
importance = model.feature_importances_   
ordered_importances = pd.DataFrame()
ordered_importances['feature'] = X_train.columns
ordered_importances['importance'] = importance 
ordered_importances = ordered_importances.sort_values(ascending=False, by=['importance'])

n = 10
some = ordered_importances.head(n)
sum(ordered_importances['importance'].head(n))

# plot importances
feature = some['feature']
feature2 = pd.factorize(feature)[0]
importance = some['importance']
plt.bar(feature2, importance)
plt.xlabel('feature')
plt.ylabel('importance')
plt.show()

# for i in feature:
#     if i!= 'country_code':
#         ale_eff = ale(X=X_train, model=model, feature=[i], grid_size=50, 
#             include_CI=False,plot=False)
#         a = ale_eff. index
#         eff = ale_eff['eff']
#         # plt.figure(figsize=(16, 8))
#         # i = 'pl_for_period_net_income_0'
#         plt.plot(a, eff)
#         plt.xlabel(i)
#         plt.ylabel('Effect on the predicted probability of default')

#         i = i+'.png'
#         plt.savefig(i)
#         plt.show()  
#         plt.close()

# 10,50,200,1000
i = 'solvency_0'       
ale_eff = ale(X=X_train, model=model, feature=[i], grid_size=10000, 
    include_CI=False,  plot=True)
a = ale_eff. index
eff = ale_eff['eff']
plt.plot(a, eff)
plt.xlabel(i)
plt.ylabel('Effect on the predicted probability of default')
i = i+'.png'
plt.savefig(i)
plt.show()  
plt.close()        


feat_eff2 = ale(X=X_train, model=model, feature=['country_code'], grid_size=50,
               include_CI=False)

feat_eff2['eff']
X_train['country_code']
X_train2.country_code.unique()
 
countries = pd.DataFrame()
countries['int']=X_train.country_code.unique()
countries['country']=X_train2.country_code.unique()
countries = countries.sort_values(ascending=True, by=['int'])

a = countries['country']
eff = feat_eff2['eff']
plt.figure(figsize=(14, 7))
plt.bar(a, eff)
plt.xlabel('country_code')
plt.ylabel('Effect on the predicted probability of default')
plt.show()
plt.savefig('country_code.png')

from lime import lime_tabular

prob_values = model.predict_proba(X_test)[:, 1]

df = pd.DataFrame()
df['p']=prob_values
df['y']=y_test
df.loc[df['y'] == 1]


explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Non-default','Default' ],
    mode='classification'
)
explanation = explainer.explain_instance(X_test.iloc[527], model.predict_proba, num_features=10)
with plt.style.context("ggplot"):
    explanation.as_pyplot_figure()
explanation.save_to_file('lime_report.html')
prob_values[527]



