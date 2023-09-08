import sys
import os
import warnings
import numpy as np
import pandas as pd
import json

# import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, Ridge
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from interpret import show
from pygam import LogisticGAM, LinearGAM
from pygam import terms, s, f
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from load_datasets import Dataset
import torch
from scipy.stats.mstats import winsorize


from typing import Dict
from dataclasses import dataclass, field
from igann import IGANN


class Plotter:
    def __init__(self, data_set_name, is_syn=False, random_state=1):
        self.data_set_name = data_set_name
        self.is_syn = is_syn
        self.random_state = random_state
        self.dataset = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_original = None
        self.transformers = None
        self.ct = None
        self.cat_cols = None
        self.scaler_dict = None
        self.task = None
        self.plot_data = PlotData()

        self.load_dataset()
        self.preprocess_data()
        self.split_data()
        self.create_directories()

    def load_dataset(self):
        self.dataset = Dataset(self.data_set_name, self.is_syn)
        self.task = self.dataset.problem
        self.X = self.dataset.X
        self.y = self.dataset.y

        self.X, self.y = shuffle(self.X, self.y, random_state=self.random_state)
        self.X_original = self.X.copy()

    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )

    def preprocess_data(self):
        transformers = [
            (
                "ohe",
                OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore", drop="if_binary"
                ),
                self.dataset.categorical_cols,
            ),
            ("num", FunctionTransformer(), self.dataset.numerical_cols),
        ]
        self.ct = ColumnTransformer(transformers=transformers, remainder="drop")
        self.ct.fit(self.X)
        self.X_original = self.X
        self.X = self.ct.transform(self.X)

        self.cat_cols = (
            self.ct.named_transformers_["ohe"].get_feature_names_out(
                self.dataset.categorical_cols
            )
            if len(self.dataset.categorical_cols) > 0
            else []
        )
        self.X = pd.DataFrame(
            self.X, columns=np.concatenate((self.cat_cols, self.dataset.numerical_cols))
        )

        self.scaler_dict = {}
        for c in self.dataset.numerical_cols:
            sx = StandardScaler()
            self.X[c] = sx.fit_transform(self.X[c].values.reshape(-1, 1))
            self.scaler_dict[c] = sx

    def evaluate_model(self, model):
        if self.task == "classification":
            y_pred = model.predict(self.X_test)
            results = accuracy_score(self.y_test, y_pred)
        else:
            y_pred = model.predict(self.X_test)
            results = mean_squared_error(self.y_test, y_pred)
        print(results)

    def create_directories(self):
        if not os.path.isdir("plots"):
            os.mkdir("plots")
        if not os.path.isdir(f"plots/{self.dataset.name}"):
            os.mkdir(f"plots/{self.dataset.name}")
        if not os.path.isdir("plot_data"):
            os.mkdir("plot_data")
        if not os.path.isdir(f"plot_data/{self.dataset.name}"):
            os.mkdir(f"plot_data/{self.dataset.name}")

    def winsorize_col(self, col, upper_limit, lower_limit):
        return winsorize(col, limits=[upper_limit, lower_limit])

    def map_y_range(self, a, b, Y):
        (a1, a2), (b1, b2) = a, b
        return [b1 + ((y - a1) * (b2 - b1) / (a2 - a1)) for y in Y]

    def make_plot_data(self, X, Y, feature_name, model_name):
        if isinstance(Y, np.ndarray):
            Y = Y.tolist()
        elif torch.is_tensor(Y):
            Y = Y.cpu().numpy().tolist()

        if isinstance(X, np.ndarray):
            X = X.tolist()
        elif torch.is_tensor(X):
            X = X.cpu().numpy().tolist()

        plot_data = {"model": model_name, "feature": feature_name, "X": X, "Y": Y}
        with open(
            "plot_data/"
            + self.data_set_name
            + "/"
            + model_name
            + "_"
            + feature_name
            + ".json",
            "w",
        ) as f:
            json.dump(plot_data, f)

    def make_plot(
        self,
        X,
        Y,
        feature_name,
        model_name,
        scale_back=True,
        scale_y=False,
        distplot=True,
        titel=True,
        x_label=True,
        y_label=True,
        ex_distplot=False,
        save_plot_data=True,
    ):
        X = np.array(X)
        Y = self.map_y_range((min(Y), max(Y)), (0, 100), Y) if scale_y else Y

        if feature_name in self.dataset.numerical_cols and scale_back:
            X = (
                self.scaler_dict[feature_name]
                .inverse_transform(X.reshape(-1, 1))
                .squeeze()
            )

        if save_plot_data:
            self.make_plot_data(X, Y, feature_name, model_name)

        if distplot:
            fig, (ax1, ax2) = plt.subplots(
                nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]}
            )
            if feature_name in self.dataset.numerical_cols:
                bins_values, _, _ = ax2.hist(
                    self.X_original[feature_name], bins=10, color="grey"
                )
            else:
                bins_values, _, _ = ax2.hist(X[feature_name], bins=10, color="grey")
            ax2.set_xlabel("Distribution")
            ax2.set_xticks([])
            ax2.set_yticks([0, max(bins_values)])
        else:
            fig, ax1 = plt.subplots(nrows=1)

        fig.set_size_inches(5, 4)
        fig.set_dpi(100)

        if model_name != "EBM":
            ax1.plot(X, Y, color="black", alpha=1)
        else:
            ax1.step(X, Y, where="post", color="black")

        if titel:
            ax1.set_title(f"Feature:{feature_name}")
        if x_label:
            ax1.set_xlabel(f"Feature value")
        if y_label:
            ax1.set_ylabel("Feature effect on model output")
        fig.tight_layout()
        plt.savefig(f"plots/{self.data_set_name}/{model_name}_shape_{feature_name}.png")
        # plt.show()
        plt.close(fig)

        if ex_distplot:
            self.create_dist_plot(X, feature_name, self.data_set_name)

    def create_dist_plot(self, X, feature_name, dataset_name):
        fig, ax1 = plt.subplots(nrows=1)
        fig.set_size_inches(4, 1)
        fig.set_dpi(100)

        if feature_name in self.dataset.numerical_cols:
            bins_values, _, _ = ax1.hist(
                self.X_original[feature_name], bins=10, color="grey"
            )
        else:
            bins_values, _, _ = ax1.hist(X[feature_name], bins=10, color="grey")
            ax1.set_xticks([])
            ax1.set_yticks([0, max(bins_values)])

        fig.savefig(f"plots/{dataset_name}/Distribution_shape_{feature_name}.pdf")
        fig.show()

    def make_one_hot_plot(
        self,
        class_zero,
        class_one,
        feature_name,
        model_name,
        distplot=True,
        title=True,
        x_label=True,
        y_label=True,
        ex_distplot=False,
    ):
        original_feature_name = feature_name.split("_")[0]
        if distplot:
            fig, (ax1, ax2) = plt.subplots(
                nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]}
            )
            if x_label:
                ax2.set_xlabel("Distribution")
            bins_values, _, _ = ax2.hist(
                self.X_original[original_feature_name], bins=2, rwidth=0.9, color="grey"
            )
            ax2.set_xticks([])
            ax2.set_yticks([0, max(bins_values)])
        else:
            fig, ax1 = plt.subplots(nrows=1)

        fig.set_size_inches(5, 4)
        fig.set_dpi(100)

        category_0 = self.X_original[original_feature_name].values.categories[0]
        category_1 = self.X_original[original_feature_name].values.categories[1]
        categories = [category_0, category_1]
        ax1.bar(
            [0, 1],
            [class_zero, class_one],
            color="gray",
            tick_label=[f"{categories[0]}", f"{categories[1]} "],
        )

        if title:
            plt.title(f'Feature:{feature_name.split("_")[0]}')
        if y_label:
            ax1.set_ylabel("Feature effect on model output")

        fig.tight_layout()
        plt.savefig(
            f'plots/{self.data_set_name}/{model_name}_onehot_{str(feature_name).replace("?", "")}.png'
        )
        # plt.show()
        plt.close()

        if ex_distplot:
            self.create_one_hot_dist_plot(feature_name, self.data_set_name)

    def create_one_hot_dist_plot(self, feature_name):
        original_feature_name = feature_name.split("_")[0]
        fig, ax1 = plt.subplots(nrows=1)
        fig.set_size_inches(5, 1)
        fig.set_dpi(100)
        bins_values, _, _ = ax1.hist(
            self.X_original[original_feature_name], bins=2, rwidth=0.9, color="grey"
        )
        ax1.set_xticks([])
        ax1.set_yticks([0, max(bins_values)])

        fig.savefig(f"plots/{self}/Distribution_onehot_{feature_name}.png")
        fig.show()

    def make_one_hot_multi_plot(
        self, model_name, distribution_plot=False, ex_distribution_plot=True
    ):
        for feature_name in self.plot_data.entries:
            position_list = np.arange(len(self.plot_data.entries[feature_name]))
            y_values = list(self.plot_data.entries[feature_name].values())
            y_list_not_given_class = [
                list(dict_element.values())[0] for dict_element in y_values
            ]
            y_list_given_class = [
                list(dict_element.values())[1] for dict_element in y_values
            ]

            if self.task == "regression":
                y_list_not_given_class = self.y_scaler.inverse_transform(
                    np.array(y_list_not_given_class).reshape((-1, 1))
                ).squeeze()
                y_list_given_class = self.y_scaler.inverse_transform(
                    np.array(y_list_given_class).reshape((-1, 1))
                ).squeeze()

            x_list = list(self.plot_data.entries[feature_name].keys())

            if distribution_plot:
                fig, (ax1, ax2) = plt.subplots(
                    nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]}
                )
                bins_values, _, _ = ax2.hist(
                    self.X_original[feature_name],
                    bins=len(x_list),
                    rwidth=0.8,
                    color="grey",
                )
                ax2.set_xlabel("Distribution")
                ax2.set_xticks([])
                ax2.set_yticks([0, max(bins_values)])
            else:
                fig, ax1 = plt.subplots()

            fig.set_size_inches(5, 4)
            fig.set_dpi(100)

            y_plot_value = []
            for i in range(len(y_values)):
                y_not_given_values = sum(
                    [
                        value
                        for index, value in enumerate(y_list_not_given_class)
                        if index != i
                    ]
                )
                y_plot_value.append((y_list_given_class[i] + y_not_given_values).item())

            ax1.bar(position_list, y_plot_value, color="gray", width=0.8)

            ax1.set_ylabel("Feature effect on model output")
            ax1.set_title(f"Feature:{feature_name}")
            ax1.set_xticks(position_list)
            ax1.set_xticklabels(x_list, rotation=90)
            fig.tight_layout()
            plt.savefig(
                f'plots/{self.data_set_name}/{model_name}_multi_onehot_{str(feature_name).replace("?", "")}.png',
                bbox_inches="tight",
            )
            # plt.show()

            if ex_distribution_plot:
                self.make_one_hot_multi_distribution_plot(
                    feature_name, self.data_set_name, x_list
                )

    def make_one_hot_multi_distribution_plot(self, feature_name, dataset_name, x_list):
        fig, ax1 = plt.subplots(nrows=1)
        bins_values, _, _ = ax1.hist(
            self.X_original[feature_name], bins=len(x_list), rwidth=0.8, color="grey"
        )

        fig.set_size_inches(5, 1)
        fig.set_dpi(100)
        ax1.set_xticks([])
        ax1.set_yticks([0, max(bins_values)])
        fig.savefig(f"plots/{dataset_name}/Distribution_onehot_{feature_name}.png")
        fig.show()

    def EBM(self):
        model_name = "EBM"
        if self.task == "classification":
            ebm = ExplainableBoostingClassifier(
                interactions=0, max_bins=512, outer_bags=8, inner_bags=4
            )
        else:
            ebm = ExplainableBoostingRegressor(
                interactions=0, max_bins=512, outer_bags=8, inner_bags=4
            )
        ebm.fit(self.X_train, self.y_train)
        self.evaluate_model(ebm)
        ebm_global = ebm.explain_global()

        for i, _ in enumerate(ebm_global.data()["names"]):
            data_names = ebm_global.data()
            feature_name = data_names["names"][i]
            shape_data = ebm_global.data(i)
            if shape_data["type"] == "interaction":
                pass
            elif shape_data["type"] == "univariate":
                original_feature_name = feature_name.split("_")[0]
                if self.X_original[original_feature_name].value_counts().size == 2:
                    self.make_one_hot_plot(
                        shape_data["scores"][0],
                        shape_data["scores"][1],
                        feature_name,
                        "EBM",
                    )
                elif feature_name.split("_")[0] not in self.dataset.numerical_cols:
                    column_name = feature_name.split("_")[0]
                    class_name = feature_name.split("_")[1]
                    not_given_class_score = shape_data["scores"][0]
                    given_class_score = shape_data["scores"][1]
                    self.plot_data.add_entry(
                        column_name,
                        class_name,
                        not_given_class_score,
                        given_class_score,
                    )
                else:
                    X_values = shape_data["names"].copy()
                    Y_values = shape_data["scores"].copy()
                    Y_values = np.r_[Y_values, Y_values[np.newaxis, -1]]

                    self.make_plot(X_values, Y_values, feature_name, model_name)

            else:
                raise ValueError(f"Unknown type {shape_data['type']}")

    def PYGAM(self):
        model_name = "PYGAM"
        # TODO: Integrate terms as parameters on model initialization
        tms = terms.TermList(
            *[
                f(i)
                if self.X.columns[i] in self.dataset.categorical_cols
                else s(i, n_splines=20, lam=0.6)
                for i in range(self.X.shape[1])
            ]
        )

        if self.task == "classification":
            PYGAM = LogisticGAM(tms)
            # PYGAM.predict = PYGAM.predict_proba
            print("classification PYGAM")
        elif self.task == "regression":
            PYGAM = LinearGAM(tms)
            print("regression PYGAM")

        PYGAM.fit(self.X_train, self.y_train)
        self.evaluate_model(PYGAM)

        plot_data = PlotData()
        for i, term in enumerate(PYGAM.terms):
            if term.isintercept:
                continue
            X_values = PYGAM.generate_X_grid(term=i)
            pdep, confi = PYGAM.partial_dependence(term=i, X=X_values, width=0.95)

            original_feature_name = self.X[self.X.columns[i]].name.split("_")[0]
            if (self.X_original[original_feature_name].value_counts().size > 2) and (
                original_feature_name in self.dataset.categorical_cols
            ):
                column_name = original_feature_name
                class_name = self.X[X.columns[i]].name.split("_")[1]
                not_given_class_score = pdep[0]
                given_class_score = pdep[-1]

                plot_data.add_datapoint(
                    column_name, class_name, not_given_class_score, given_class_score
                )

            if len(self.X[self.X.columns[i]].unique()) == 2:
                self.make_one_hot_plot(pdep[0], pdep[-1], self.X.columns[i], model_name)
            else:
                self.make_plot(
                    X_values[:, i].squeeze(), pdep, self.X.columns[i], model_name
                )

        self.make_one_hot_multi_plot(plot_data, model_name)

    def LR(self):
        model_name = "LR"
        if self.task == "regression":
            LR = Ridge()
        else:
            LR = LogisticRegression()
        LR.fit(self.X, self.y)
        plot_data = PlotData()
        word_to_coef = dict(zip(LR.feature_names_in_, LR.coef_.squeeze()))
        dict(sorted(word_to_coef.items(), key=lambda item: item[1]))
        word_to_coef_df = pd.DataFrame.from_dict(word_to_coef, orient="index")

        for i, feature_name in enumerate(self.X.columns):
            original_feature_name = feature_name.split("_")[0]
            if original_feature_name in self.dataset.categorical_cols:
                if self.X_original[original_feature_name].value_counts().size > 2:
                    column_name = original_feature_name
                    class_name = feature_name.split("_")[1]
                    class_score = word_to_coef[feature_name]
                    plot_data.add_datapoint(column_name, class_name, 0, class_score)
                else:
                    self.make_one_hot_plot(
                        0, word_to_coef[feature_name], feature_name, model_name
                    )  # zero as value for class one correct?
            else:
                inp = torch.linspace(
                    self.X[feature_name].min(), self.X[feature_name].max(), 1000
                )
                outp = word_to_coef[feature_name] * inp
                # convert back to list before plooting.
                inp = inp.cpu().numpy().tolist()
                outp = outp.cpu().numpy().tolist()
                self.make_plot(inp, outp, feature_name, model_name)
        self.make_one_hot_multi_plot(plot_data, model_name)

    def IGANN(self):
        model_name = "IGANN"
        igann = IGANN(self.task, n_estimators=1000, device="cpu")
        igann.fit(self.X, np.array(self.y))

        plot_data = PlotData()
        shape_data = igann.get_shape_functions_as_dict()
        for feature in shape_data:
            original_feature_name = feature["name"].split("_")[0]
            if original_feature_name in self.dataset.categorical_cols:
                if self.X_original[original_feature_name].value_counts().size > 2:
                    # print(feature)
                    column_name = original_feature_name
                    class_name = feature["name"].split("_")[1]
                    not_given_category_value = feature["y"].numpy()[0]
                    if len(feature["y"].numpy()) == 2:
                        given_category_value = feature["y"].numpy()[1]
                    elif len(feature["y"].numpy()) == 1:
                        given_category_value = 0
                    else:
                        raise ValueError(
                            "Feature has neither than 2 nor 1 value. This should not be possible."
                        )
                    plot_data.add_datapoint(
                        column_name,
                        class_name,
                        not_given_category_value,
                        given_category_value,
                    )
                else:
                    self.make_one_hot_plot(
                        feature["y"][0],
                        feature["y"][1],
                        feature["name"],
                        model_name,
                    )
            else:
                self.make_plot(
                    feature["x"],
                    feature["y"],
                    feature["name"],
                    model_name,
                )

        self.make_one_hot_multi_plot(plot_data, model_name)


class PlotData:
    def __init__(self):
        self.entries = {}

    def add_entry(
        self, feature_name, class_name, not_given_class_score, given_class_score
    ):
        if feature_name not in self.entries:
            self.entries[feature_name] = {}
        self.entries[feature_name][class_name] = {
            "not_given_class_score": not_given_class_score,
            "given_class_score": given_class_score,
        }
