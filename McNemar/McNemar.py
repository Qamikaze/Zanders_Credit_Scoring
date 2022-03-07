#####
# This file is used to perform the McNemar's test to compare the performance of different classification models.
#####

from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
from numpy import genfromtxt
pred_values_model1 = genfromtxt('...pred_lr.csv', delimiter=',',skip_header=1)#Load in the predicted values (binary, so 1 for default, 0 for non-default) for model 1
pred_values_model2 = genfromtxt('...pred_lr.csv', delimiter=',',skip_header=1) #Load in the predicted values (binary, so 1 for default, 0 for non-default) for model 2
y_test = genfromtxt('...y_test.csv', delimiter=',',skip_header=1)  #Load in the true values which need to be predicted
model1_correctness = (pred_values_model1==y_test)
model2_correctness = (pred_values_model2==y_test)

#Create a contingency table
contingency_table = np.zeros((2,2))
for i in range(0,len(pred_values_model1)):
    #if y_test[i]==1: #Uncomment this line when you want to apply the test solely on the default cases
        if model1_correctness[i]==True and model2_correctness[i]==True:
            contingency_table[0,0] += 1
        elif model1_correctness[i]==False and model2_correctness[i]==False:
            contingency_table[1,1] += 1
        elif model1_correctness[i]==True and model2_correctness[i]==False:
            contingency_table[0,1] += 1
        elif model1_correctness[i]==False and model2_correctness[i]==True:
            contingency_table[1,0] += 1
        
result = mcnemar(contingency_table, exact=False,correction=False)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')
