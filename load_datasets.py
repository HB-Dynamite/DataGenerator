import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder


class Dataset:
    def __init__(self, name: str, is_syn=False, biased=False):
        self.name = name
        """ name of the dataset"""
        self.is_syn = is_syn
        """ boolean to handle synthetic datasets"""
        self.biased = biased
        """ bool to decide for biased or unbiased version"""
        self.problem = None
        """ classification or regression """
        self.X = None
        """ X data frame """
        self.y = None
        """ y data frame """
        self.labels = None
        """ discrete label values of the dataset: classification: [0, 1]"""
        self.target_names = None
        """ name of the label: classification: ['Negative Class', 'Positive Class'], regression: ['ValueName'] """
        self.numerical_cols = None
        """ list of numerical columns which are selected after preprocessing"""
        self.categorical_cols = None
        """ list of categorical feature names which are selected after preprocessing """
        self.basic_dataset_metadata = None
        """ dictionary with basic dataset metadata """
        self.preprocessing_dataset_metadata = None
        """ dictionary with preprocessing dataset metadata """

        # Now load all these variables
        self._load_by_name()
        self._replace_underscore()

    def _replace_underscore(self):
        self.X.columns = [col.replace("_", " ") for col in self.X.columns]
        self.numerical_cols = [col.replace("_", " ") for col in self.numerical_cols]
        self.categorical_cols = [col.replace("_", " ") for col in self.categorical_cols]

    def _load_by_name(self):
        self.load_data()

    def _preprocess_columns(self, df):
        """
        Drop variables with missing values >50%.
        Replace missing numerical variable values by mean.
        Replace missing categorical variable values by -1.
        Drop categorical columns with more than 25 distinct values.
        :return:
        """

        assert (
            len(self.numerical_cols) + len(self.categorical_cols) > 0
        ), "Dataframe columns must be specified in load_datasets.py in order to preprocess them."

        # Ensure there are no empty string in numerical columns and encode as float
        df.loc[:, self.numerical_cols] = df.loc[:, self.numerical_cols].replace(
            {"": np.nan, " ": np.nan}
        )

        # select columns with more than 50 % missing values
        incomplete_cols = df.columns[df.isnull().sum() / len(df) > 0.5]
        # select categorical_cols with more than 25 unique values
        detailed_cols = (
            df[self.categorical_cols]
            .nunique()[df[self.categorical_cols].nunique() > 25]
            .index.tolist()
        )

        self.numerical_cols = list(set(self.numerical_cols) - set(incomplete_cols))
        self.categorical_cols = list(
            set(self.categorical_cols) - set(incomplete_cols) - set(detailed_cols)
        )

        df = df.loc[:, self.numerical_cols + self.categorical_cols]

        # For categorical columns with values: fill n/a-values with -1.
        print("cat_cols2:", self.categorical_cols)
        if len(self.categorical_cols) > 0:
            for categorical_col in self.categorical_cols:
                df[categorical_col] = df[categorical_col].fillna("unknown")
                df[categorical_col] = df[categorical_col].astype("category")

        # For numerical columns with values: fill n/a-values with mean.
        if len(self.numerical_cols) > 0:
            for num_col in self.numerical_cols:
                df.loc[:, num_col] = pd.to_numeric(df.loc[:, num_col], errors="coerce")
                df.loc[:, num_col] = df.loc[:, num_col].fillna(
                    df.loc[:, num_col].median()
                )

        return df

    def load_data(self):
        # 22
        df = pd.read_csv("data/" + self.name + ".csv", sep=",")
        self.basic_dataset_metadata = {
            "n_samples": df.shape[0],
            "n_columns": df.shape[1],
        }
        self.categorical_cols = []
        target_col = [col for col in df.columns if col.startswith("target_")]
        biased_target_col = [col for col in df.columns if col.startswith("biased_")]

        feature_cols = [
            col for col in df.columns if col not in target_col + biased_target_col
        ]
        print(f"target: {target_col}")
        print(f"biased target: {biased_target_col}")
        print(f"features: {feature_cols}")

        for col in feature_cols:
            if df[col].nunique() < 10:
                self.categorical_cols.append(col)
        print(f"cat_cols:{self.categorical_cols}")
        self.numerical_cols = [
            col for col in feature_cols if col not in self.categorical_cols
        ]

        if self.biased:
            self.y = df.loc[:, biased_target_col]
        else:
            self.y = df.loc[:, target_col]

        self.y = pd.DataFrame(self.y)

        unique_values = self.y.nunique().values
        print(unique_values)
        if unique_values > 2:
            self.problem = "regression"
        else:
            self.problem = "classification"
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
            print("encoded Labels:", self.y)

        self.X = self._preprocess_columns(df)

        self.target_names = None
        self.preprocessing_dataset_metadata = {
            "n_samples": self.X.shape[0],
            "n_columns": self.X.shape[1] + 1,
        }
