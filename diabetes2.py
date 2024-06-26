# %%
import numpy as np
from pprint import pprint
import os
from DataGen import DataGenerator
from plotter import Plotter
from load_datasets import Dataset
import gamchanger as gc


# Initialize the Data Generator with 10,000 observations
ds_generator = DataGenerator(10000, "diabetes")

# %%
# Add Gender variable using categories
ds_generator.add_var(
    name="gender", categories=["male", "female"], base_probs=[0.5, 0.5]
)

# %%
# Add Age variable using a normal distribution
ds_generator.add_var(
    name="age", distribution="normal", dist_params={"loc": 40, "scale": 10}
)

# %%
# Add Height variable using a conditional distribution based on gender
ds_generator.add_var(
    name="height",
    categorical_var="gender",
    dist_dict={
        "male": {"dist": "normal", "params": {"loc": 175, "scale": 8}},
        "female": {"dist": "normal", "params": {"loc": 165, "scale": 7}},
    },
)

# %%
# Add Weight variable using an expression
ds_generator.add_var(
    name="weight",
    expression="(1 - 0.1*('gender' == 'female')) * height - 100",
    noise_level=0.1,
)

# %%
# Add BMI variable using an expression
ds_generator.add_var(name="bmi", expression="weight / (height / 100) ** 2")

# %%
# Create Diabetes target variable
ds_generator.add_target(
    categories=["yes", "no"],
    name="diabetes",
    base_probs=[0.05, 0.95],  # TODO: find better way to integrate base probs are
    exp_dict={
        "yes": "0.6*math.exp(bmi-24) + 0.1*(age-54)",
        "no": "1",
    },  # TODO: find better way to calc probs from expressions
    exp_level=1,  # no influence of base probs
    noise_level=0.1,
)

# %%
# Add a bias to increase the probability of diabetes for males
# TODO: it should be more cleare how the target variable gets biased.
# 1. Target should always beeing the main porb giver for the target (consider a factor)
# 2. It may be cool if the bias is more observations more that where at the
ds_generator.add_bias(
    categories=["yes", "no"],
    base_probs=[0.05, 0.95],
    exp_dict={
        "yes": "0.5*(target_diabetes == 'yes') + 0.5*(gender == 'male')",
        "no": "0.5*(target_diabetes == 'no')+ 0.5*(gender == 'female')",
    },
    exp_level=1,  # no base probs. Soley rely on other vars
    noise_level=0.05,
)

# %%
# try missing functions
ds_generator.explore_data(vars=["age"])
ds_generator.add_missing_at_random("age", 0.1, "max(0,bmi-20)")
ds_generator.explore_data(vars=["age"], dataset_mode="missing")

# ds_generator.add_missing_complety_at_random("age", 0.3)


# %%
ds_generator.hide_var("height")
ds_generator.hide_var("weight")

# %%
# Save the dataset as a CSV file
ds_generator.save_as_csv()
ds_generator.save_as_csv(dataset_mode="unbiased")
ds_generator.save_as_csv(dataset_mode="missing")

# %%
# Output metadata
pprint(ds_generator.get_metadata())

# %% create network graph
ds_generator.generate_graph()

# %%
ds_generator.explore_data()


# %% init Plotter
diabetes_biased = Dataset("diabetes_biased", biased=True)
print(diabetes_biased)
plotter_biased = Plotter(dataset=diabetes_biased)  # will load data by name

# run models and get shapeplots (Plots will be saved as png)
# %%
### EBM ###
plotter_biased.EBM()

# %%
### PYGAM ###
plotter_biased.PYGAM()

# %%
### IGANN ###
plotter_biased.IGANN()

# %%
### LR ###
plotter_biased.LR()


# %% init Plotter
# plotter = Plotter("diabetes",is_syn = False) # example for gamCompare datasets
diabetes_unbiased = Dataset("diabetes_unbiased", biased=False)
plotter_unbiased = Plotter(dataset=diabetes_unbiased)  # will load data by name

# run models and get shapeplots (Plots will be saved as png)
# %%
### EBM ###
plotter_unbiased.EBM()

# %%
### PYGAM ###
plotter_unbiased.PYGAM()

# %%
### IGANN ###
plotter_unbiased.IGANN()

# %%
### LR ###
plotter_unbiased.LR()


# %%
# inti gamchanger for biased data
gc.visualize(
    plotter_biased.model_dict["ebm"],
    plotter_biased.X_test,
    plotter_biased.y_test,
)


# %%
# inti gamchanger for unbiased data
gc.visualize(
    plotter_unbiased.model_dict["ebm"],
    plotter_unbiased.X_test,
    plotter_unbiased.y_test,
)


# %%
# inti gamchanger for model from biased data on unbiased data
gc.visualize(
    plotter_biased.model_dict["ebm"],
    plotter_unbiased.X_test,
    plotter_unbiased.y_test,
)

# %%
print(os.getcwd())
