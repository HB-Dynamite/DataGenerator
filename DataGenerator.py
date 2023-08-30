# %%
import numpy as np
from pprint import pprint
import pandas as pd
import os
import math
import re  # To parse the expression string


class DatasetGenerator:
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
        expressions=None,
        exp_level=None,
        categorical_var=None,
        dist_dict=None,
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
            }

        elif distribution:
            data = self.gen_from_dist(distribution, dist_params)
            self.dataset[name] = self.add_noise(data, noise_level)

            metadata = {
                "type": "distribution",
                "distribution": distribution,
                "dist_params": dist_params,
                "noise_level": noise_level,
            }

        elif (
            categorical_var and dist_dict
        ):  # Numeric variable based on a categorical variable
            data = self.gen_numeric_from_cat(categorical_var, dist_dict)
            self.dataset[name] = self.add_noise(data, noise_level)

            metadata = {
                "type": "numeric_from_cat",
                "categorical_var": categorical_var,
                "dist_map": dist_dict,
                "noise_level": noise_level,
            }

        elif categories:
            if (
                expressions and exp_level is not None
            ):  # Conditional categorical variable from expressions
                data = self.gen_conditional_categorical_var_from_exp(
                    categories, base_probs, expressions, exp_level
                )
                self.dataset[name] = self.add_categorical_noise(
                    data, noise_level, categories
                )

                metadata = {
                    "type": "conditional_categorical_from_exp",
                    "categories": categories,
                    "base_probs": base_probs,
                    "expressions": expressions,
                    "exp_level": exp_level,
                    "input_vars": [
                        self.extract_var_from_re(expression)
                        for expression in expressions
                    ],
                    "noise_level": noise_level,
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
        print(size)
        size = self.n_observations if size is None else size
        print(size)

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

    def gen_conditional_categorical_var_from_exp(
        self, categories, base_probs, expressions, exp_level
    ):
        """
        Add a conditional categorical variable to the dataset based on an input feature.

        Args:
        - name (str): Name of the new variable.
        - base_on (list): List of names of the input features used in the expression.
        - categories (list): List of categories.
        - base_probs (list): Base probabilities for each category.
        - expressions (str): Expressions to compute probabilities adaptions for each category. higher values mean higher probability for this category
            - keep in mind that the probalities are transformed with a soft max function to ensure the sum of probs is 1.

        Returns:
        - data for new var
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

        def chose_categorie(row):
            exp_probs = calculate_probs_from_exp(row, expressions)
            final_probs = combine_base_and_exp_probs(base_probs, exp_probs, exp_level)
            # chose catgore based on final probs
            chosen_category = np.random.choice(categories, p=final_probs)
            return chosen_category

        # Generate data
        data = self.dataset.apply(chose_categorie, axis=1).to_numpy()

        return data

    def gen_numeric_from_cat(self, categorical_var, dist_map):
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

    def remove_var(self, name):
        self.dataset = self.dataset.drop(name, axis=1)
        return self

    def get_dataset(self):
        return self.dataset

    def get_metadata(self):
        return self.metadata

    def get_name(self):
        return self.name

    def save_as_csv(self, file_name=None):
        if file_name is None:
            file_name = self.name + ".csv"
        path = os.path.join("data", "synthetic", file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.dataset.to_csv(path, index=False)

    def save_alt_as_csv(self, file_name=None):
        if file_name is None:
            file_name = self.name + "_alt" + ".csv"
        path = os.path.join("data", "synthetic", file_name)
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


# %%
# 1. Create a new DatasetGenerator instance with 1000 observations named "my_dataset"
data_gen = DatasetGenerator(1000, "my_dataset")

# 2. Add variables x1, x2, etc., using chaining

data_gen.add_var(
    "x1", distribution="normal", dist_params={"loc": 5, "scale": 2}
).add_var("x2", distribution="uniform", dist_params={"low": 0, "high": 10}).add_var(
    "x3", expression="x1 + x2", noise_level=0.5
)

# 3. Check the dataset
# print(data_gen.get_dataset().head())

# 4. Save the dataset as CSV
data_gen.save_as_csv()


# For further additions, just continue chaining:

data_gen.add_var(
    "x4", distribution="normal", dist_params={"loc": 3, "scale": 1}
).add_var("x5", expression="x4*2")


data_gen.add_var(
    "cat1",
    categories=["n0", "n1"],
    base_probs=[0.5, 0.5],
    expressions=["x1", "-x2"],
    exp_level=0.5,
    noise_level=0.3,
)
# 6. Check the updated dataset
print(data_gen.get_dataset().head(100))

pprint(data_gen.get_metadata())

# %%
# Assuming DatasetGenerator class is defined above this code

# Create the dataset generator with 1000 observations
ds_generator = DatasetGenerator(1000, "medical_dataset")

# Gender: Categorical with two categories ('male', 'female')
ds_generator.add_var(
    name="gender", categories=["male", "female"], base_probs=[0.5, 0.5]
)

# Age: Normally distributed, mean age 40, standard deviation 10
ds_generator.add_var(
    name="age", distribution="normal", dist_params={"loc": 40, "scale": 10}
)

# Height: Normally distributed but different for each gender.
# For males, mean height is 175 cm, standard deviation is 8.
# For females, mean height is 165 cm, standard deviation is 7.
ds_generator.add_var(
    name="height",
    categorical_var="gender",
    dist_dict={
        "male": {"dist": "normal", "params": {"loc": 175, "scale": 8}},
        "female": {"dist": "normal", "params": {"loc": 165, "scale": 7}},
    },
)

# Weight: Normally distributed but dependent on height.
# Weight = 0.9 * height - 100 + noise
ds_generator.add_var(name="weight", expression="0.9 * height - 100", noise_level=0.05)

# BMI: Calculated as weight / (height/100)^2
ds_generator.add_var(name="bmi", expression="weight / (height / 100) ** 2")

# Insurance: Categorical, either 'yes' or 'no' with different probabilities
ds_generator.add_var(name="insurance", categories=["yes", "no"], base_probs=[0.7, 0.3])

# Diabetes: Categorical ('yes', 'no') but conditionally dependent on BMI and age.
# The higher the BMI and age, the higher the chance of having diabetes.
ds_generator.add_var(
    name="diabetes",
    categories=["yes", "no"],
    base_probs=[0.1, 0.9],
    expressions=["bmi * 0.05 + age * 0.05", "1"],
    exp_level=0.5,
)

# Risk of diabetes: A numeric variable that is a function of age, bmi, and whether the person has insurance.
# Calculated as (0.3 * age + 0.5 * bmi - 5 * (insurance == 'yes'))
ds_generator.add_var(
    name="risk_of_diabetes",
    expression="0.3 * age + 0.5 * bmi - 5 * (insurance == 'yes')",
    noise_level=0.1,
)

# Save the dataset
ds_generator.save_as_csv("medical_dataset.csv")

# Output metadata
print("Dataset metadata:")
pprint(ds_generator.get_metadata())
