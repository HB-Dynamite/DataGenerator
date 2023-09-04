import numpy as np
from pprint import pprint
import pandas as pd
import os
import math
import re  # To parse the expression string
from typing import List, Dict, Union


class DataGenerator:
    def __init__(self, n_observations, name):
        self.n_observations = n_observations
        self.name = name
        self.dataset = pd.DataFrame()
        self.metadata = {}

    def add_noise(self, data, noise_level):
        """
        Add normal distributed noise to the data based on its data mean.

        :param data: Original data array.
        :param noise_level: Proportional level of noise.
                            - 0: No noise
                            - 1: Noise magnitude roughly equivalent to the mean
        :return: Data with noise added.
        """
        # not sure if this is the best method to fit the scale of noise to data values.
        # Idea here is to add use mean so that data of small magniute get small noise and vise versa.
        sd = np.mean(data) * noise_level
        noise = np.random.normal(0, sd, size=len(data))
        return data + noise

    def add_categorical_noise(self, data, noise_level, categories):
        """
        Add noise to categorical data.

        :param data: Original categorical data array.
        :param noise_level: Proportion of the data to be modified (0-1).
        :param categories: List of categories to choose from.
        :return: Data with noise added.
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
        Add a new variable to the dataset.
        Depending on the provided arguments, the variable can be:
        - Generated from a mathematical expression
        - Generated from a specified distribution
        - Generated as a categorical variable
        - Generated as a conditional categorical variable based on one expression per categorie
        """

        if expression:
            data = self.gen_from_exp(expression)
            self.dataset[name] = self.add_noise(data, noise_level)

            metadata = {
                "type": "expression",
                "expression": expression,
                "noise_level": noise_level,
                "input_vars": self.extract_var_from_re(expression),
                "lvl_measurement": "numeric"
                if lvl_measurment is None
                else lvl_measurment,
            }

        elif distribution:
            data = self.gen_from_dist(distribution, dist_params)
            self.dataset[name] = self.add_noise(data, noise_level)

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
        ):  # Numeric variable based on a categorical variable
            data = self.gen_numeric_from_cat_dist(categorical_var, dist_dict)
            self.dataset[name] = self.add_noise(data, noise_level)

            metadata = {
                "type": "numeric_from_cat_dist",
                "categorical_var": categorical_var,
                "dist_map": dist_dict,
                "noise_level": noise_level,
                "lvl_measurement": "numeric"
                if lvl_measurment is None
                else lvl_measurment,
            }
        elif categorical_var and exp_dict:
            data = self.gen_numeric_from_cat_exp(categorical_var, exp_dict)
            self.dataset[name] = self.add_noise(data, noise_level)

            metadata = {
                "type": "numeric_from_cat_exp",
                "categorical_var": categorical_var,
                "exp_map": exp_dict,
                "noise_level": noise_level,
                "lvl_measurement": "numeric"
                if lvl_measurment is None
                else lvl_measurment,
            }

        elif categories:
            if (
                exp_dict and exp_level is not None
            ):  # Conditional categorical variable from expressions
                data = self.gen_categorical_var_from_exp(
                    categories, base_probs, exp_dict, exp_level
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
                    "input_vars": [
                        self.extract_var_from_re(expression)
                        for expression in list(exp_dict.values())
                    ],
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
        category_exp_map: Dict[str, str],
        base_probs: List[float],
        exp_level: float,
    ) -> np.ndarray:
        """
        Generate a conditional categorical variable based on given features.

        Args:
            category_expr_map (Dict[str, str]): Mapping from categories to expressions.
            base_probs (List[float]): Base probabilities for each category.
            exp_level (float): A level for balancing the influence of base_probs and adjusted probabilities.

        Returns:
            np.ndarray: The generated categorical data.
        """

        def softmax(x):
            e_x = np.exp(x - np.max(x))  # Subtract np.max(x) for numerical stability
            return e_x / e_x.sum(axis=0, keepdims=True)

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
            return softmax(np.array(probs))

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

        categories = list(category_exp_map.keys())
        expressions = list(category_exp_map.values())

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

    def create_target(self, name, **kwargs):
        """
        Simple method to create a target variable.
        This methods simply calles the add_var method but prefixes the var name with _target.
        Args:
        name: name of target variable str
        **kwargs: kwargs for add var
        """
        self.add_var("target_" + name, **kwargs)

    def add_bias(self, **kwargs):
        """
        Simple method do introduce a bias into the target var.
        Simple creates a new var that is prefixed with bias.
        TODO: should ensure that the orignial target is part of the input variables.(complicated check)
        """
        target_name = self.get_target_var_name()
        biased_target_name = "biased_" + target_name
        self.add_var(biased_target_name, kwargs)

    def get_dataset(self):
        return self.dataset

    def get_metadata(self):
        return self.metadata

    def get_name(self):
        return self.name

    def save_as_csv(self, file_name=None):
        if file_name is None:
            file_name = self.name + ".csv"
        path = os.path.join("data", file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.dataset.to_csv(path, index=False)

    def save_alt_as_csv(self, file_name=None):
        if file_name is None:
            file_name = self.name + "_alt" + ".csv"
        path = os.path.join("data", file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        alt_dataset = self.get_alt_dataset()
        alt_dataset.to_csv(path, index=False)

    def extract_var_from_re(self, expression):
        # Extract potential variable names from the expression
        variables = re.findall("[A-Za-z_][A-Za-z0-9_]*", expression)

        # Filter out any math functions/constants from the list of extracted names
        math_attributes = set(dir(math))
        filtered_variables = [var for var in variables if var not in math_attributes]

        return filtered_variables

    def get_target_var_name(self):
        for var_name, var_metadata in self.metadata.items():
            if var_name.startswith("target_"):
                return var_name
        raise Exception("No target variable found in metadata.")
