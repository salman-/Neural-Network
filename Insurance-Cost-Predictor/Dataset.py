import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self):
        self.TEST_TRAIN_PORTION = 0.2
        self.TARGET_COLUMN = "charges"
        self.data = pd.read_csv("./insurance.csv")

        all_features_except_target = self.get_all_features_except_target()
        target_feature = self.get_target()

        self.train_x, self.test_x, \
        self.train_y, self.test_y = \
            self.get_train_test_datasets(all_features_except_target, target_feature)

    def process_data(self, x):
        column_transformer = self.get_column_transformer()
        column_transformer.fit(X=x)
        return column_transformer.transform(x)

    def get_column_transformer(self):
        return make_column_transformer(
            (MinMaxScaler(), ["age", "bmi", "children"]),
            (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
        )

    def get_train_test_datasets(self, x, target):
        return train_test_split(x, target, test_size=self.TEST_TRAIN_PORTION, random_state=42)

    def get_all_features_except_target(self):
        return self.data.loc[:, self.data.columns != self.TARGET_COLUMN]

    def get_target(self):
        return self.data[self.TARGET_COLUMN]
