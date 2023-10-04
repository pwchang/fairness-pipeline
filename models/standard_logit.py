from sklearn.linear_model import LogisticRegression

class StandardLogit:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        self.model = LogisticRegression(penalty="none", solver = "lbfgs", max_iter=1000, random_state=109)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict_proba(X_test)

    def recommend(self, X_test):
        return self.model.predict(X_test)