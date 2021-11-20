from sklearn.datasets import make_circles
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self):

        n_sample = 1000
        X, y = make_circles(n_sample, noise=0.03, random_state=42)
        self.dt = pd.DataFrame({"Coordinate_X": X[:, 0], "Coordinate_Y": X[:, 1], "Label": y})

        self.train, self.test, self.label_train, self.label_test = \
            train_test_split(self.dt.loc[:, self.dt.columns != "Label"], self.dt["Label"], train_size=0.8)

    def create_category_label(self, label, first_category_name, second_category_name):
        return label.apply(self.find_correct_category, args=(first_category_name, second_category_name)).values.tolist()

    def find_correct_category(self, x, first_category_name, second_category_name):
        return first_category_name if x else second_category_name
