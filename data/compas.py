import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from random import choices

class Compas:
    def __init__(self, filename):
        print('Reading in: ', filename)
        df = pd.read_csv(filename)

        print('Processing data')
        df['sex'] = (df['sex'] == 'Male') * 1
        df['felony'] = (df['c_charge_degree'] == 'F') * 1

        one_hot_df = pd.get_dummies(df['race'], prefix='race', drop_first=False)
        df = pd.concat([df, one_hot_df], axis=1)
        df = df.drop(['c_charge_degree', 'c_charge_desc', 'score_text', 'race_Other','priors_count',
                                'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'decile_score',
                                'length_of_stay', 'length_of_stay_thresh'], axis=1)
        self.train_df, self.test_df = train_test_split(df, train_size=0.8, stratify=df['race'],
                                                       random_state=109)
        self.df = df.drop(['race'], axis=1)
        self.train_df = self.train_df.drop(['race'], axis=1)
        self.test_df = self.test_df.drop(['race'], axis=1)

        self.target = 'two_year_recid'
        X_train, X_test = self.train_df.drop(columns=self.target), self.test_df.drop(columns=self.target)
        self.y_train, self.y_test = np.array(self.train_df[self.target]), np.array(self.test_df[self.target])

        scaler = MinMaxScaler().fit(X_train)
        self.X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(X_test)

    # Example use: column=race_African-American and trait = 1
    def get_idx(self, column, trait):
        idx = np.where(self.test_df[column]==trait)[0]
        return idx

    # increased sampling of a target subgroup
    # d=.5 means 50% increase in sampling
    def differential_sampling(self, column, trait, d=.5):
        idx_test = np.where((self.test_df[column] == trait) & (self.test_df[self.target]==1))[0]
        idx_train = np.where((self.train_df[column] == trait) & (self.train_df[self.target]==1))[0]

        test_add = choices(idx_test, k=round(d*len(idx_test)))
        train_add = choices(idx_train, k=round(d * len(idx_train)))

        X_test_add, y_test_add, X_train_add, y_train_add = [], [], [], []
        for id in test_add:
            X_test_add.append(self.X_test[id])
            y_test_add.append(self.y_test[id])
        for id in train_add:
            X_train_add.append(self.X_train[id])
            y_train_add.append(self.y_train[id])

        # TODO: implement noise function

        self.X_test = np.r_[self.X_test, X_test_add]
        self.X_train = np.r_[self.X_train, X_train_add]
        self.y_test = np.concatenate((self.y_test, y_test_add))
        self.y_train = np.concatenate((self.y_train, y_train_add))
