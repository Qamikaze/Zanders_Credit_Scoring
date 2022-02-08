import pandas as pd
from math import log
from matplotlib import pyplot
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors  import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

start_time = datetime.now()

# Load raw data
data = pd.read_csv('output_file.gzip', compression='gzip', header=0, sep=',', quotechar='"')
names = data.columns 

# Get label 
y = data['default_indicator']
defaults = y.sum()
non_defaults = len(y)-y.sum()
default_rate = defaults/len(y)

# Check missing values
missing = data.isnull().sum()/len(data)*100
sorted_missing = missing.sort_values()
total_missing = sum(sorted_missing.head(n=108))/108

defaults_part = data.loc[data['default_indicator'] == 1]
missing2 = defaults_part.isnull().sum()/len(defaults_part)*100
sorted_missing2 = missing2.sort_values()

# Get covariates
x = data.drop(columns = sorted_missing.tail(n=4).index.values)
x = x.drop(columns=['index', 'random_id', 'default_indicator'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
X_train = pd.concat([X_train,X_test], ignore_index=True)


names_var = ['intangible_fixed_assets','current_assets','debtors',
             'total_assets','shareholders_funds','noncurrent_liabilities',
             'current_liabilities','loans','creditors','operating_revenue',
             'costs_of_goods_sold','gross_profit','operating_pl_ebit',
             'pl_for_period_net_income','interest_paid','ebitda','capital_employed',
             'liquidity','total_debt','tangible_net_worth','turnover_growth',
             'gross_margin','operating_margin','return_on_sales','return_on_capital_employed',
             'current_ratio','quick_ratio','stock_days','debtor_days','creditor_days',
             'gearing','solvency','debt_ebitda_ratio','interest_coverage_ratio']

start_time = datetime.now()

pd.set_option('mode.chained_assignment',None)

def imputer(var):

    data_set_train = X_train.loc[:, X_train.columns.str.startswith(var)]
    data_set_test = X_test.loc[:, X_test.columns.str.startswith(var)]
    print(var)
         
    # Impute missing values
    imputer_model = KNeighborsRegressor(n_neighbors=15)
    imputer = IterativeImputer(max_iter=1, random_state=0, estimator=imputer_model)
    imputer.fit(data_set_train)
    
    # Transform training data
    train_data = imputer.transform(data_set_train)
    train_data = pd.DataFrame(data=train_data, columns=data_set_train.columns)
    for i in data_set_train.columns:
        X_train[i]=train_data[i].values
    

    # Transform testing data
    test_data = imputer.transform(data_set_test)
    test_data = pd.DataFrame(data=test_data, columns=data_set_test.columns)
    for i in data_set_test.columns:
        X_test[i]=test_data[i].values
for var in names_var:
    imputer(var)
imputer('stock')    

    
# Create train set
X_train['y'] = y_test
X_train.to_csv("train_data.csv.gz", index=False, compression="gzip")

# Create test set
X_test['y'] = y_test
X_test.to_csv("test_data.csv.gz", index=False, compression="gzip")

print('Duration: {}'.format(datetime.now() - start_time))
