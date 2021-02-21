import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        self.trim()
        self.validate()
        self.base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        np.random.seed(2)

        n = len(self.df)

        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]
        
        self.df_train = df_shuffled.iloc[:n_train].copy()
        self.df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        self.df_test = df_shuffled.iloc[n_train+n_val:].copy()

        self.y_train_orig = self.df_train.msrp.values
        self.y_val_orig = self.df_val.msrp.values
        self.y_test_orig = self.df_test.msrp.values

        self.y_train = np.log1p(self.df_train.msrp.values)
        self.y_val = np.log1p(self.df_val.msrp.values)
        self.y_test = np.log1p(self.df_test.msrp.values)

        del self.df_train['msrp']
        del self.df_val['msrp']
        del self.df_test['msrp']
        
        pass

    def prepare_X(self, df):
        df_num = df[self.base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

    def linear_regression(self):
        X_train = self.prepare_X(self.df_train)
        y = self.y_train
        ones = np.ones(X_train.shape[0])
        X = np.column_stack([ones, X_train])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        biased_term = w[0]
        weights = w[1:]

        y_pred = biased_term + X_train.dot(weights)
        y_pred = np.round(np.expm1(y_pred), 0)
        desired_columns = ['engine_cylinders', 'transmission_type', 'driven_wheels', 'number_of_doors', 
        'market_category', 'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity']
        output = self.df_train[desired_columns]
        output['msrp'] = self.y_train_orig
        output['msrp_pred'] = y_pred
        print(output.head(5).to_markdown())

        # pass

def test():
    car_price = CarPrice()
    car_price.linear_regression()
    
if __name__ == "__main__":
    test()