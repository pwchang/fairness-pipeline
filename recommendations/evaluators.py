import numpy as np
from sklearn.metrics import confusion_matrix

# Functions to evaluate bias in predictions
# idx is the indices of the subgroup we are looking at. If blank, it evaluates whole dataset
def performance_metrics(model, X_test, y_test, idx=[]):
    if len(idx)==0:
        idx = np.arange(len(y_test))
    X_test = X_test[idx]
    y_test = y_test[idx]

    y_pred = model.predict(X_test)
    model_accuracy = model.score(X_test, y_test)

    confusion = dict(zip(['tn', 'fp', 'fn', 'tp'], confusion_matrix(y_test, y_pred).ravel()))
    fpr = confusion['fp'] / (confusion['fp'] + confusion['tn'])
    fnr = confusion['fn'] / (confusion['fn'] + confusion['tp'])
    precision = confusion['tp'] / (confusion['fp'] + confusion['tp'])
    recall = confusion['tp'] / (confusion['fn'] + confusion['tp'])

    return dict(zip(['model_accuracy', 'fpr', 'fnr', 'precision', 'recall'],
                    [model_accuracy, fpr, fnr, precision, recall]))

# insufficiently justified disparate impact. We want ijdi to be positive
def ijdi(model, X_test, y_test, idx, lam=10):
    not_idx = np.setdiff1d(np.arange(len(y_test)), idx)
    fpr_S = performance_metrics(model, X_test, y_test, idx)['fpr']
    fpr_notS = performance_metrics(model, X_test, y_test, not_idx)['fpr']

    preds_S = model.predict_proba(X_test[idx])
    p_S = sum([p[0] for p in preds_S])/len(idx)
    preds_notS = model.predict_proba(X_test[not_idx])
    p_notS = sum([p[0] for p in preds_notS])/len(not_idx)

    ijdi = lam*(p_S-p_notS) - (fpr_S-fpr_notS)
    return ijdi

# FPR etc comes from the recommendations i.e. the predicted 1s and 0s
# base rate comes from the sum of the predicted probabilities from the model
# idji measures whether the recommendation mechanism (e.g. threshold) is biased