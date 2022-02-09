import pandas as pd
from math import log
from matplotlib import pyplot
from datetime import datetime
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from hmeasure import h_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


start_time = datetime.now()

def validation_splitter():
    # Load raw data
    train = pd.read_csv('train_data.csv.gz', compression='gzip',  header=0, sep=',', quotechar='"')
    
    # Factorize data
    train['country_code'] = pd.factorize(train['country_code'])[0]
    train['industry_code'] = pd.factorize(train['industry_code'])[0]
    train['size_class'] = pd.factorize(train['size_class'])[0]
    train['status_year'] = pd.factorize(train['status_year'])[0]
    
    # Seperate labels and covariates
    X = train.drop(['y'], axis=1)
    y = train['y']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=15/85, random_state=42, stratify=y)

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

    return X_train[c], y_train, X_val[c], y_val

def get_data(small):
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
        X, y, test_size=15/85, random_state=42, stratify=y)
    
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


# =============================================================================
# Performance metrics
# =============================================================================

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




X_trainMM, y_trainMM, X_testMM, y_testMM = get_data(False)


# =============================================================================
# Weight of Evidence transformations
# =============================================================================

names = X_trainMM.columns 
categorical_names = ['country_code','industry_code','size_class','status_year']
numerical_names = names.drop(categorical_names)
defaults = y_trainMM.sum()
non_defaults = len(y_trainMM)-y_trainMM.sum()

small_data = X_trainMM
small_data['y'] = y_trainMM
woe_data = small_data
def WoE_plotter(x):
    weights = []
    values = small_data[x].unique()
    
    for v in values:
        print(v)
        subset = small_data[small_data[x]== v]
        y = subset['y'] 
        WoE =  ((len(y) - sum(y)) / non_defaults) / (sum(y)/defaults)
        if sum(y)!=0:
            WoE= log(WoE)
        else:
            WoE= log((((len(y) - sum(y))+0.5)/non_defaults)/((sum(y)+0.5)/defaults))
        weights.append(WoE)
    
    df= pd.DataFrame()
    df['values'] = values
    df['weights'] = weights
    df.sort_values(by=['weights'])
    
    df['weights'].plot()
    df2= df.sort_values(by=['weights'])
    df2= df2.reset_index(drop=True)
    df2.plot()
    pyplot.xlabel("Int")
    pyplot.ylabel("WoE")
    # print(df)
    # print(df2)
    return df2
    
def numeric_WoE_plotter(x):
    dataset = small_data[['y',x]]
    dataset = dataset.sort_values(x)
    weights = []
    values = list(range(1, 21))
    size = int(len(dataset)/20)

    A = -99999999999999
    
    for v in values:
        B = size*v
        subset = dataset.iloc[A:B]
        y = subset['y'] 
        WoE =  ((len(y) - sum(y)) / non_defaults) / (sum(y)/defaults)
        if sum(y)!=0:
            WoE= log(WoE)
        else:
            WoE= log((((len(y) - sum(y))+0.5)/non_defaults)/((sum(y)+0.5)/defaults))
        weights.append(WoE)
        A = B
        
    df= pd.DataFrame()
    df['values'] = values
    df['weights'] = weights
    
    df['weights'].plot()
    df2= df.sort_values(by=['weights'])
    df2= df2.reset_index(drop=True)
    df2['weights'].plot()
    pyplot.xlabel("Int")
    pyplot.ylabel("WoE")
    # print(df)
    # print(df2)
    return df2
    
for column_name in categorical_names:
    df = WoE_plotter(column_name)
    df_dict = dict(zip(df['values'],df.index.values))
    woe_data[column_name] = woe_data[column_name].replace(df_dict)
    X_testMM[column_name] = X_testMM[column_name].replace(df_dict)
    
number = 0
for column_name in numerical_names:
    df = numeric_WoE_plotter(column_name)
    df_dict = dict(zip(df['values'],df.index.values))
    size = int(len(small_data)/20)
    woe_data = woe_data.sort_values(column_name)
    for key,value in sorted(df_dict.items()):
        A = size*(key-1)
        B = size*key
        if A==0:
            start = -999999999999
            end = woe_data.iloc[B][column_name]
        elif B==size*20:
            start = woe_data.iloc[A][column_name]
            end = 9999999999999
        else:
            start = woe_data.iloc[A][column_name]
            end = woe_data.iloc[B][column_name]
        X_testMM[column_name] = X_testMM[column_name].mask((start < X_testMM[column_name]) & (X_testMM[column_name]< end), other = value)
        woe_data.loc[woe_data.index[A:B],column_name] = value
    number += 1
    print(number)


woe_data = woe_data.sort_index()

number = 0
for column_name in numerical_names:
    df = numeric_WoE_plotter(column_name)
    df_dict = dict(zip(df['values'],df.index.values))
    size = int(len(small_data)/20)
    woe_data = woe_data.sort_values(column_name)
    for key,value in sorted(df_dict.items()):
        A = size*(key-1)
        B = size*key
        if A==0:
            start = -999999999999
            end = woe_data.iloc[B][column_name]
        elif B==size*20:
            start = woe_data.iloc[A][column_name]
            end = 9999999999999
        else:
            start = woe_data.iloc[A][column_name]
            end = woe_data.iloc[B][column_name]
        X_testMM[column_name] = X_testMM[column_name].mask((start < X_testMM[column_name]) & (X_testMM[column_name]< end), other = value)
        woe_data.loc[woe_data.index[A:B],column_name] = value
    number += 1
    print(number)


woe_data = woe_data.sort_index()

X_train_woe = woe_data.drop(['y'],axis=1)
y_train_woe = woe_data['y']

print('Duration: {}'.format(datetime.now() - start_time))

# =============================================================================
# General to specific setup
# =============================================================================

start_time = datetime.now()
logit_model=sm.Logit(y_train_woe,X_train_woe)
result=logit_model.fit(method='bfgs',maxiter=1000)
pvalues = result.pvalues
maxValue = max(pvalues)
i= 0
while maxValue > 0.05:
    print(maxValue)
    maxIndex = pvalues.idxmax()
    print(maxIndex)
    X_train_woe = X_train_woe.drop([maxIndex],axis=1)
    logit_model=sm.Logit(y_train_woe,X_train_woe)
    result=logit_model.fit(method='bfgs')
    pvalues = result.pvalues
    maxValue = max(pvalues)
    i += 1
    print(i)
result.summary()
print('Duration: {}'.format(datetime.now() - start_time))

# Fit LR with reduced variables
start_time = datetime.now()

c = X_train_woe.columns.values.tolist()

# c = ['country_code', 'industry_code', 'size_class',
#        'intangible_fixed_assets_2', 'pl_for_period_net_income_0',
#        'interest_paid_2', 'ebitda_0', 'ebitda_2', 'tangible_net_worth_2',
#        'turnover_growth_0', 'operating_margin_0', 'operating_margin_2',
#        'return_on_sales_2', 'return_on_capital_employed_0', 'quick_ratio_0',
#        'stock_days_0', 'stock_days_1', 'debtor_days_1', 'creditor_days_1',
#        'gearing_0', 'gearing_1', 'gearing_2', 'solvency_0', 'solvency_1',
#        'solvency_2', 'debt_ebitda_ratio_0', 'debt_ebitda_ratio_2',
#        'interest_coverage_ratio_0']

lr = LogisticRegression(solver='saga', random_state=0, 
                        max_iter=1000, n_jobs=-1, class_weight= 'balanced')
model = lr.fit(X_train_woe[c], y_train_woe)


# Get test performance
pred_values = model.predict(X_testMM[c])
prob_values = model.predict_proba(X_testMM[c])[:, 1]
get_performance(y_testMM, pred_values, prob_values)

print('Duration: {}'.format(datetime.now() - start_time))
