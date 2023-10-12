### Script for running pipeline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import data
import models
import recommendations

# Each section should have evaluators and bias generators

# Data stage
print('Reading in data')
Data = data.Compas('input/compas.csv')

# Data bias
# TODO: implement basic bias (concept shift, reporting bias) within each module
#Data.differential_sampling('race_Caucasian',1,d=.5)

# Model stage
Model = models.StandardLogit(Data.X_train, Data.y_train)
Model.fit()
# TODO: evaluators- generalization? Model error?

# Prediction stage
#print(Model.predict(Data.X_test[0:10]))
# TODO: implement unconditional calibration evaluator
# TODO: implement conditional calibration evaluator


# Recommendations stage
#print(Model.recommend(Data.X_test[0:10]))
aa_idx = Data.get_idx('race_African-American',1)
hi_idx = Data.get_idx('race_Hispanic',1)
cc_idx = Data.get_idx('race_Caucasian',1)
print('IJDI')
print('White: ', recommendations.ijdi(Model.model, Data.X_test, Data.y_test, cc_idx))
print('Black: ', recommendations.ijdi(Model.model, Data.X_test, Data.y_test, aa_idx))
print('Hispanic: ', recommendations.ijdi(Model.model, Data.X_test, Data.y_test, hi_idx))
# TODO: incorporate IJDI-scan
# TODO: implement UJDI evaluator
# Basic confusion matrix evaluator
# print('Performance Metrics')
# print('All: ', recommendations.performance_metrics(Model.model, Data.X_test, Data.y_test))
# print('White: ', recommendations.performance_metrics(Model.model, Data.X_test, Data.y_test, cc_idx))
# print('Black: ', recommendations.performance_metrics(Model.model, Data.X_test, Data.y_test, aa_idx))
# print('Hispanic: ', recommendations.performance_metrics(Model.model, Data.X_test, Data.y_test, hi_idx))


# Decisions stage


# Impacts stage


# Outcomes stage
