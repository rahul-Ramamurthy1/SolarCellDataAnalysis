import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import optimize
from pycaret.regression import *

#Reading in and replacing columns
data = pd.read_csv("C:\\Users\\rahul\\Downloads\\Plant_1_Generation_Data.csv")
new_dates = []
for i in range(0,len(data)):
    new_dates.append(i*0.714)
data['DATE_TIME'] = new_dates


#Regression
s = setup(data,target='TOTAL_YIELD',fold_shuffle=True,imputation_type='iterative')
best = compare_models()
evaluate_model(best)
preds = predict_model(best,data=data)
save_model(best,'my_best_model')








