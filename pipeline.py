### Script for running pipeline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import data
import models


# Data stage
print('Reading in data')
data = data.Compas('input/compas.csv')

# Data bias
# TODO: implement basic bias (concept shift, reporting bias) within each module

# Model stage
model = models.StandardLogit(data.X_train, data.y_train)
model.fit()

# Prediction stage
print(model.predictions(data.X_test[0:10]))


# Recommendations stage
print(model.recommendations(data.X_test[0:10]))


# Decisions stage


# Impacts stage


# Outcomes stage
