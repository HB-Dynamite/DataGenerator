import numpy as np
from pprint import pprint
import pandas as pd
import os
import math
import re  # To parse the expression string
from typing import List, Dict, Union
import warnings

import networkx as nx
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, n_observations, name):
        """
        Parameters:
        - n_observations (int): The number of observations to be generated.
        - name (str): The name of the DataGenerator object.

        Attributes:
        - n_observations (int): Stores the number of observations.
        - name (str): Stores the name of the DataSet.
        - dataset (pd.DataFrame): An empty DataFrame to store the generated data.
        - metadata (dict): An empty dictionary to store metadata about the variables.
        """
        self.n_observations = n_observations
        self.name = name
        self.dataset = pd.DataFrame()
        self.metadata = {}

    def add_numerical_noise(self, data, noise_level):
        """
        Adds Gaussian (normal) noise to the given data array based on its mean.

        Parameters:
        - data (array-like): The original data array to which noise will be added.
        - noise_level (float): The proportional level of noise to add. Ranges from 0 to 1 ( higher is possible).
                            - 0: No noise
                            - 1: Noise magnitude roughly equivalent to the mean of the data

        Returns:
        - array-like: The data array with added noise.

        Note:
        - The method uses the mean of the data to scale the noise. Therefore, the noise_level
        parameter controls the magnitude of the noise relative to the data mean.
        """
        # not sure if this is the best method to fit the scale of noise to data values.
        # Idea here is to add use mean so that data of small magniute get small noise and vise versa.
        sd = np.mean(data) * noise_level
        noise = np.random.normal(0, sd, size=len(data))
        return data + noise

    def add_categorical_noise(self, data, noise_level, categories):
        """
        Adds noise to a given categorical data array.

        Parameters:
        - data (array-like): The original categorical data array.
        - noise_level (float): The proportion of the data to be modified. Ranges from 0 to 1.
                            - 0: No noise
                            - 1: All data points may be modified
        - categories (list): A list of categories to choose from when adding noise.

        Returns:
        - array-like: The data array with added noise.

        Note:
        - The method randomly selects a subset of the data based on the noise_level and
        replaces their categories with random choices from the provided categories list.
        """
        # Check if there's a need for noise
        if noise_level <= 0:
            return data

        # Determine how many entries to modify
        n_noise = int(noise_level * self.n_observations)

        # Choose random indices to change
        noise_indices = np.random.choice(
            self.n_observations, size=n_noise, replace=False
        )

        # Replace with a random category for those indices
        data[noise_indices] = np.random.choice(categories, size=n_noise)

        return data

    def add_var(
        self,
        name,
        expression=None,
        distribution=None,
        dist_params=None,
        noise_level=0,
        categories=None,
        base_probs=None,
        exp_level=None,
        categorical_var=None,
        dist_dict=None,
        exp_dict=None,
        lvl_measurment=None,
    ):
        """
        Adds a new variable to the dataset based on various options.

        Parameters:
        - name (str): The name of the new variable.
        - expression (str, optional): A mathematical expression to generate the variable.
        - distribution (str, optional): The name of the distribution to use for generating the variable.
        - dist_params (dict, optional): Parameters for the specified distribution.
        - noise_level (float, optional): The level of noise to add. Default is 0.
        - categories (list, optional): A list of categories for a categorical variable.
        - base_probs (list, optional): Base probabilities for each category.
        - exp_level (float, optional): A level for balancing the influence of base_probs and adjusted probabilities.
        - categorical_var (str, optional): Name of an existing categorical variable to base the new variable on.
        - dist_dict (dict, optional): A dictionary mapping categories to distributions for a conditional variable.
        - exp_dict (dict, optional): A dictionary mapping categories to expressions for a conditional variable.
        - lvl_measurment (str, optional): The level of measurement for the variable ('numeric' or 'categorical').

        Returns:
        - DataGenerator: Returns the DataGenerator object for method chaining.

        Note:
        - The method is highly flexible and can generate variables based on expressions, distributions,
        or existing categorical variables. The type of variable generated depends on the arguments provided.
        """

        if expression:
            data = self.gen_from_exp(expression)
            self.dataset[name] = self.add_numerical_noise(data, noise_level)

            metadata = {
                "type": "expression",
                "expression": expression,
                "noise_level": noise_level,
                "input_vars": self.get_input_vars(expression),
                "lvl_measurement": "numeric"
                if lvl_measurment is None
                else lvl_measurment,
            }

        elif distribution:
            data = self.gen_from_dist(distribution, dist_params)
            self.dataset[name] = self.add_numerical_noise(data, noise_level)

            metadata = {
                "type": "distribution",
                "distribution": distribution,
                "dist_params": dist_params,
                "noise_level": noise_level,
                "lvl_measurement": "numeric"
                if lvl_measurment is None
                else lvl_measurment,
            }

        elif (
            categorical_var and dist_dict
        ):  # Numeric variable based on distributions for categories
            data = self.gen_numeric_from_cat_dist(categorical_var, dist_dict)
            self.dataset[name] = self.add_numerical_noise(data, noise_level)

            metadata = {
                "type": "numeric_from_cat_dist",
                "categorical_var": categorical_var,
                "dist_map": dist_dict,
                "noise_level": noise_level,
                "lvl_measurement": "numeric"
                if lvl_measurment is None
                else lvl_measurment,
                "input_vars": [categorical_var],
            }
        elif categorical_var and exp_dict:
            # Numeric from expressions for categroies
            data = self.gen_numeric_from_cat_exp(categorical_var, exp_dict)
            self.dataset[name] = self.add_numerical_noise(data, noise_level)
            input_vars = []
            for expression in list(exp_dict.values()):
                exp_input_vars = self.extract_var_from_re(expression)
                for var in exp_input_vars:
                    input_vars.append(var)
            unique_input_vars = set(input_vars)

            metadata = {
                "type": "numeric_from_cat_exp",
                "categorical_var": categorical_var,
                "exp_map": exp_dict,
                "noise_level": noise_level,
                "input_vars": self.get_input_vars(exp_dict.values()),
                "lvl_measurement": "numeric"
                if lvl_measurment is None
                else lvl_measurment,
            }

        elif categories:
            if (
                exp_dict and exp_level is not None
            ):  # Conditional categorical variable from expressions
                data = self.gen_categorical_var_from_exp(
                    categories=categories,
                    base_probs=base_probs,
                    exp_dict=exp_dict,
                    exp_level=exp_level,
                )
                self.dataset[name] = self.add_categorical_noise(
                    data, noise_level, categories
                )

                metadata = {
                    "type": "conditional_categorical_from_exp",
                    "categories": categories,
                    "base_probs": base_probs,
                    "exp_dict": exp_dict,
                    "exp_level": exp_level,
                    "input_vars": self.get_input_vars(exp_dict.values()),
                    "noise_level": noise_level,
                    "lvl_measurement": "categorical"
                    if lvl_measurment is None
                    else lvl_measurment,
                }

            else:  # Simple categorical variable
                data = self.gen_categorical_var(categories, base_probs, noise_level)
                self.dataset[name] = self.add_categorical_noise(
                    data, noise_level, categories
                )
                metadata = {
                    "type": "simpe categorical",
                    "categories": categories,
                    "base_probs": base_probs,
                    "noise_level": noise_level,
                    "lvl_measurement": "categorical"
                    if lvl_measurment is None
                    else lvl_measurment,
                }

        else:
            raise ValueError("Not enough information provided to generate variable.")

        # Save metadata for this variable
        self.metadata[name] = metadata

        return self

    def gen_from_dist(self, distribution, dist_params, size=None):
        """
        Generate data based on a given statistical distribution and its parameters.

        Args:
        - distribution (str): Name of the distribution (e.g., "uniform", "normal").
        - dist_params (dict): Parameters for the distribution.
        - size (int): Number of data points to generate.

        Returns:
        - np.array: Data generated from the distribution.
        """
        size = self.n_observations if size is None else size

        distributions = {
            "uniform": (np.random.uniform, {"low": 0, "high": 1}),
            "normal": (np.random.normal, {"loc": 0, "scale": 1}),
            # add more distributions
        }

        if distribution not in distributions:
            raise ValueError(f"Unsupported distribution: {distribution}.")

        dist_func, params = distributions[distribution]
        params.update(dist_params or {})
        pprint(params)
        data = dist_func(size=size, **params)

        return data

    def gen_categorical_var(self, categories, probs=None, noise_level=0):
        """
        Add a categorical variable to the dataset.

        Args:
        - name (str): Name of the variable.
        - categories (list): List of categories.
        - probs (list): Probabilities associated with each category.
                        If None, uniform distribution over categories is assumed.
        - noise_level (float): Fraction of entries for which the category will be randomly changed.

        Returns:
        - self for method chaining.
        """

        # If no probabilities are provided, assume uniform distribution
        if probs is None:
            probs = [1 / len(categories) for _ in categories]
        # TODO: check if prob sum is 1 and force it to

        # Generate data from categorical distribution
        data = np.random.choice(categories, size=self.n_observations, p=probs)

        return data

    def gen_from_exp(self, expression):
        """
        Generate data based on a provided mathematical expression.

        Args:
        - expression (str): Mathematical expression to evaluate.
        - data_frame (pd.DataFrame): Data frame used as context for the expression.

        Returns:
        - np.array: Data generated from the expression.
        """

        def eval_expression(row):
            try:
                return eval(expression, {"math": math}, row)
            except Exception as e:
                print(f"Error evaluating expression '{expression}' for row {row}: {e}")
                return 0  # Default to zero if there's an error

        data = self.dataset.apply(eval_expression, axis=1).to_numpy()

        return data

    def gen_categorical_var_from_exp(
        self,
        categories: List[str],
        exp_dict: Dict[str, str],
        base_probs: List[float],
        exp_level: float,
    ) -> np.ndarray:
        """
        Generate a conditional categorical variable based on given features.

        Args:
            categories: List of possbile catgories for the new var
            category_expr_map (Dict[str, str]): Mapping from categories to expressions.
            base_probs (List[float]): Base probabilities for each category.
            exp_level (float): A level for balancing the influence of base_probs and adjusted probabilities.

        Returns:
            np.ndarray: The generated categorical data.
        """

        def norm(x):
            x_min = x.min()
            if x_min < 0:
                warnings.warn(
                    "Expression evaluated to negative probability. Check if this was intended"
                )
            x = x - x_min
            return x / x.sum()

        def calculate_probs_from_exp(row, expressions):
            probs = []
            for expression in expressions:
                try:
                    result = eval(expression, {"math": math}, row)
                except Exception as e:
                    print(
                        f"Error evaluating expression '{expression}' for row {row}: {e}"
                    )
                    result = 0  # Default to zero if there's an error
                probs.append(result)
            # ensure porbs sum up to 1 and return
            return norm(np.array(probs))

        def combine_base_and_exp_probs(base_probs, exp_probs, exp_level):
            return (
                np.array(base_probs) * (1 - exp_level) + np.array(exp_probs) * exp_level
            )

        def chose_categorie(row: pd.Series) -> str:
            """
            Choose a category based on final probabilities.

            Args:
                row (pd.Series): A single row from the DataFrame.

            Returns:
                str: The chosen category.
            """
            exp_probs = calculate_probs_from_exp(row, expressions)
            final_probs = combine_base_and_exp_probs(base_probs, exp_probs, exp_level)
            # chose catgore based on final probs
            chosen_category = np.random.choice(categories, p=final_probs)
            return chosen_category

        categories = list(exp_dict.keys())
        expressions = list(exp_dict.values())

        # Generate data
        data = self.dataset.apply(chose_categorie, axis=1).to_numpy()

        return data

    def gen_numeric_from_cat_dist(self, categorical_var, dist_map):
        """
        Generate a new numerical column based on categories in another column.
        for each cat a new distribution is created with specified parameters
        Args:
        - name (str): Name of the new numerical column.
        - categorical_var (str): Name of the existing categorical column.
        - dist_map (dict): A dictionary mapping categories to distributions and their params.

        Returns:
        - None: Adds the new column to the DataFrame in-place.
        """

        unique_categories = self.dataset[categorical_var].unique()
        # init data with 0
        data = np.zeros(self.n_observations)

        for cat in unique_categories:
            # get subset of dataframe that is of cat
            mask = self.dataset[categorical_var] == cat
            subset_size = mask.sum()

            # get dist info from dict
            dist_info = dist_map.get(cat, None)
            if dist_info is None:
                raise ValueError(f"No distribution found for category {cat}")

            distribution = dist_info["dist"]
            params = dist_info["params"]
            print(subset_size)
            # create numeric variable based on distribution
            generated_data = self.gen_from_dist(distribution, params, size=subset_size)
            data[mask] = generated_data
        return data

    def gen_numeric_from_cat_exp(self, categorical_var, expression_map):
        unique_catgories = self.dataset[categorical_var].unique()
        data = np.zeros(self.n_observations)
        for cat in unique_catgories:
            # get subset of dataframe this is of categorie
            mask = self.dataset[categorical_var] == cat
            subset_size = mask.sum()
            # get expression for cat
            expression = expression_map[cat]
            if expression is None:
                raise ValueError(f"No expression found for the catgorie {cat}!")
            else:
                # gerate data from respective expression
                generated_data = self.gen_from_exp(expression)
            data[mask] = generated_data
        return data

    def remove_var(self, name):
        self.dataset = self.dataset.drop(name, axis=1)
        return self

    def hide_var(self, name):
        self.metadata[name].update({"hidden": True})
        return self

    def add_target(self, name, **kwargs):
        """
        Simple method to create a target variable.
        This methods simply calles the add_var method but prefixes the var name with _target.
        Args:
        name: name of target variable str
        **kwargs: kwargs for add var
        """
        target_name = "target_" + name
        self.add_var(target_name, **kwargs)
        self.metadata[target_name].update({"target": True})

    def add_bias(self, **kwargs):
        """
        Simple method do introduce a bias into the target var.
        Simple creates a new var that is prefixed with bias.
        TODO: should ensure that the orignial target is part of the input variables.(complicated check)
        """
        target_name = self.get_target_names(biased=False)[0]
        print(target_name)
        biased_target_name = target_name + "_biased"
        self.add_var(biased_target_name, **kwargs)
        self.metadata[biased_target_name].update({"biased": True, "target": True})

    def get_dataset(self, mode="biased"):
        mode_dict = {
            "full": self.dataset,
            "biased": self.dataset.loc[
                :, self.get_not_hidden_vars_names() + self.get_target_names(biased=True)
            ],
            "unbiased": self.dataset.loc[
                :,
                self.get_not_hidden_vars_names() + self.get_target_names(biased=False),
            ],
        }
        return mode_dict[mode]

    def get_X(self, mode="not_hidden"):
        mode_dict = {
            "full": self.dataset.loc[:, self.get_var_names()],
            "not_hidden": self.dataset.loc[:, self.get_not_hidden_vars_names()],
        }
        return mode_dict[mode]

    def get_y(self, mode="biased"):
        return self.dataset.loc[:, self.get_target_names(biased=(mode == "biased"))]

    def get_hidden_vars_names(self):
        hidden_vars = []
        for var in self.metadata.keys():
            if (
                "hidden" in self.metadata[var].keys()
                and self.metadata[var]["hidden"] == True
            ):
                hidden_vars.append(var)
        return hidden_vars

    def get_var_names(self):
        vars = []
        for var in self.metadata.keys():
            var_metadata = self.metadata[var]
            if "target" not in var_metadata:
                vars.append(var)
        return vars

    def get_not_hidden_vars_names(self):
        not_hidden_vars = []
        for var in self.get_var_names():
            var_metadata = self.metadata[var]
            if (
                "hidden" in self.metadata[var].keys()
                and self.metadata[var]["hidden"] == False
            ) or ("hidden" not in self.metadata[var]):
                not_hidden_vars.append(var)
        return not_hidden_vars

    def get_target_names(self, biased=False):
        targets = []
        for var, var_metadata in self.metadata.items():
            if "target" in var_metadata:
                # If looking for biased targets
                if biased and var_metadata.get("biased") == True:
                    targets.append(var)
                # If looking for unbiased targets
                elif not biased and not var_metadata.get("biased"):
                    targets.append(var)
        return targets

    def get_metadata(self):
        return self.metadata

    def get_name(self):
        return self.name

    def save_as_csv(self, file_name=None, mode="biased"):
        if file_name is None:
            file_name = self.name + "_" + mode + ".csv"
        path = os.path.join("data", file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.get_dataset(mode=mode).to_csv(path, index=False)

    def extract_var_from_re(self, expression):
        # Extract potential variable names from the expression
        variables = re.findall("[A-Za-z_][A-Za-z0-9_]*", expression)

        # Filter out any math functions/constants from the list of extracted names
        math_attributes = set(dir(math))
        filtered_variables = [var for var in variables if var not in math_attributes]

        return filtered_variables

    def get_input_vars(self, expressions):
        # Ensure expressions is a list
        if isinstance(expressions, str):
            expressions = [expressions]
        print(expressions)
        input_vars = []
        for expression in expressions:
            exp_input_vars = self.extract_var_from_re(expression)
            for var in exp_input_vars:
                print(var)
                print(self.dataset.columns)
                if var in self.dataset.columns:
                    input_vars.append(var)
        unique_input_vars = set(input_vars)
        return unique_input_vars

    def generate_graph(self):
        G = nx.DiGraph()
        nodes = list(self.metadata.keys())
        for node in nodes:
            G.add_node(node)
            if "input_vars" in self.metadata[node].keys():
                for input_var in self.metadata[node]["input_vars"]:
                    if input_var in nodes:  # ensure the input_var is a valid node
                        G.add_edge(input_var, node)

        def generate_x_positions(metadata):
            layers = []
            remaining_nodes = set(metadata.keys())
            while remaining_nodes:
                layer = {
                    node
                    for node in remaining_nodes
                    if all(
                        dep not in remaining_nodes
                        for dep in metadata[node].get("input_vars", [])
                    )
                }
                if not layer:
                    raise ValueError(
                        "No valid layer found; graph may have cycles or missing nodes"
                    )
                layers.append(layer)
                remaining_nodes -= layer

            x_pos = {}
            for x, layer in enumerate(layers):
                for node in layer:
                    x_pos[node] = x
            return x_pos

        # Generate y-positions using NetworkX's spring_layout
        pos = nx.spring_layout(G)

        # Generate custom x-positions
        x_pos = generate_x_positions(self.metadata)

        # Update x-coordinates while keeping y-coordinates from spring_layout
        for node in pos:
            pos[node][0] = x_pos.get(node, 0)

        nx.draw(
            G,
            pos,  # Pass the positions here
            with_labels=True,
            node_color="lightblue",
            font_weight="bold",
            node_size=700,
            font_size=10,
        )
        plt.show()

    def explore_data(self):
        """
        This function allows a quick peek into the generated data.
        """

        def plt_hists(vars):
            for var in vars:
                data = self.dataset[var]
                plt.hist(data, bins=30, color="skyblue", edgecolor="black", alpha=0.7)

                plt.title(f"Histogram of {var}")
                plt.xlabel(var)
                plt.ylabel("Frequency")
                plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)

                plt.show()
                plt.close()

        plt_hists(self.dataset.columns)
