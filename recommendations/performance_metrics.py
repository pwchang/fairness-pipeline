import numpy as np
from sklearn.metrics import confusion_matrix

# Functions to evaluate bias in predictions
# idx is the indices of the subgroup we are looking at. If blank, it evaluates whole dataset
def performance_metrics(model, X_test, y_true, idx=[]):
    if len(idx)==0:
        idx = np.arange(len(y_true))
    X_test = X_test[idx]
    #y_true = y_true.values[idx]
    y_true = y_true[idx]

    y_pred = model.predict(X_test)
    model_accuracy = model.score(X_test, y_true)

    confusion = dict(zip(['tn', 'fp', 'fn', 'tp'], confusion_matrix(y_true, y_pred).ravel()))
    fpr = confusion['fp'] / (confusion['fp'] + confusion['tn'])
    fnr = confusion['fn'] / (confusion['fn'] + confusion['tp'])
    precision = confusion['tp'] / (confusion['fp'] + confusion['tp'])
    recall = confusion['tp'] / (confusion['fn'] + confusion['tp'])

    return dict(zip(['model_accuracy', 'fpr', 'fnr', 'precision', 'recall'],
                    [model_accuracy, fpr, fnr, precision, recall]))
