#####
# This file performs a variable selection by removing highly correlated features.
#####

import pandas as pd
from datetime import datetime

# Load raw data
x = pd.read_csv('train_data.csv.gz', compression='gzip',  header=0, sep=',', quotechar='"')
train = x
y = train['y']
train = train.drop(['y'], axis=1)

train['country_code'] = pd.factorize(train['country_code'])[0]
train['industry_code'] = pd.factorize(train['industry_code'])[0]
train['size_class'] = pd.factorize(train['size_class'])[0]
train['status_year'] = pd.factorize(train['status_year'])[0]

start_time = datetime.now()

# Use correlations to reduce the bumber of features
l = train.columns.values.tolist()
for i in train.columns:
    if i in l:
        for j in train.columns:
            corr = train[i].corr(train[j])
            if corr > 0.7 and j!=i:
                print(corr)
                if y.corr(train[i]) >  y.corr(train[j]):
                    print('dropping ', j)
                    print( 'correlated with', i)
                    train = train.drop(columns=[j])
                    l.remove(j)
                else:
                    print('dropping ', i)
                    print( 'correlated with', j)
                    train = train.drop(columns=[i])
                    l.remove(i)
                    break

corrMatrix = train.corr()     
      
print('Duration: {}'.format(datetime.now() - start_time)) 


