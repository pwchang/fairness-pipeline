from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

class StandardLogit:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.scaler = MinMaxScaler().fit(self.X_train)

    def fit(self):
        X_train_scaled = self.scaler.transform(self.X_train)
        self.model = LogisticRegression(penalty="none", solver = "lbfgs", max_iter=1000, random_state=109)
        self.model.fit(X_train_scaled, self.y_train)

    def predictions(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)

    def recommendations(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)