import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Compas:
    def __init__(self, filename):
        print('Reading in: ', filename)
        self.df = pd.read_csv(filename)

        print('Processing data')
        self.df['sex'] = (self.df['sex'] == 'Male') * 1
        self.df['felony'] = (self.df['c_charge_degree'] == 'F') * 1

        one_hot_df = pd.get_dummies(self.df['race'], prefix='race', drop_first=False)
        compas_race_df = pd.concat([self.df.drop('race', axis=1), one_hot_df], axis=1)
        self.compas_race_df = compas_race_df.drop(['c_charge_degree', 'c_charge_desc', 'score_text', 'race_Other'], axis=1)
        self.train_df, self.test_df = train_test_split(self.compas_race_df, train_size=0.8, stratify=self.df['race'],
                                             random_state=109)

        self.df.drop(columns=['priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'decile_score',
                                'length_of_stay', 'length_of_stay_thresh'])
        X_drop = ['priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                  'decile_score', 'two_year_recid', 'length_of_stay', 'length_of_stay_thresh']
        self.X_train, self.X_test = self.train_df.drop(columns=X_drop), self.test_df.drop(columns=X_drop)
        self.y_train, self.y_test = self.train_df['two_year_recid'], self.test_df['two_year_recid']

        scaler = MinMaxScaler().fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    # Example use: aa_idx = np.where(compas_race_df['race_African-American']==1)[0]
    def get_idx(self, column, trait):
        idx = np.where(self.test_df[column]==trait)[0]
        return idx