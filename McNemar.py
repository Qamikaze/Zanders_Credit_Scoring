# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:10:28 2022

@author: svenh
"""

from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
pred_values_model1 = pred_values #Load in the predicted values (binary) for model 1
pred_values_model2 = pred_values_2 #Load in the predicted values (binary) for model 2
y_test = y_test #Load in the y_test model
model1_correctness = (pred_values_model1==y_test)
model2_correctness = (pred_values_model2==y_test)

contingency_table = np.zeros((2,2))
for i in range(0,len(pred_values_model1)):
	#if y_test[i]==1: #Dit begin uncomment als we willen testen op alleen de default cases
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
