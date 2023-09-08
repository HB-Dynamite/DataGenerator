# %%
import numpy as np
from pprint import pprint
from DataGen import DataGenerator
from plotter import Plotter


# Initialize the Data Generator with 100,000 observations
ds_generator = DataGenerator(100000, "diabetes")

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
ds_generator.create_target(
    categories=["yes", "no"],
    name="diabetes",
    base_probs=[0.05, 0.95],
    exp_dict={"yes": "0.05*bmi+ 0.05*age", "no": "1"},
    exp_level=0.7,
    noise_level=0.1,
)

# %%
# Add a bias to increase the probability of diabetes for males
ds_generator.add_bias(
    categories=["yes", "no"],
    base_probs=[0.05, 0.95],
    exp_dict={
        "yes": "0.9*('target_diabetes' == 'yes') + 0.1*(gender == 'male')",
        "no": "0.9*('target_diabetes' == 'no') - 0.1*('gender' == 'male')",
    },
    exp_level=0.7,
    noise_level=0.1,
)

# %%
ds_generator.remove_var("height")
ds_generator.remove_var("weight")

# %%
# Save the dataset as a CSV file
ds_generator.save_as_csv()

# Output metadata
pprint(ds_generator.get_metadata())

# %% create network graph
ds_generator.generate_graph()

# %%
# ds_generator.explore_data()


# %% init Plotter
# plotter = Plotter("diabetes",is_syn = False) # example for gamCompare datasets
plotter = Plotter("diabetes", is_syn=True)  # will load data by name

# run models and get shapeplots (Plots will be saved as png)
# %%
### EBM ###
plotter.EBM()

# %%
### PYGAM ###
plotter.PYGAM()

# %%
### IGANN ###
plotter.IGANN()

# %%
### LR ###
plotter.LR()
# %%
