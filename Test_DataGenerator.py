# %%
import unittest
import os

from DataGenerator import DatasetGenerator


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = DatasetGenerator(100, "test_dataset")

        # Adding variables based on your requirements
        self.generator.add_var(
            "X1", distribution="normal", dist_params={"loc": 0, "scale": 1}
        )
        self.generator.add_var(
            "X2", distribution="uniform", dist_params={"low": 0, "high": 1}
        )
        self.generator.add_var("X3", expression="X1 + X2")

    def test_var_normal(self):
        dataset = self.generator.get_dataset()
        # check if normal dist var exists
        self.assertTrue("X1" in dataset.columns)
        # check if the first entry of the is a float or an int
        self.assertTrue(isinstance(dataset["X1"].iloc[0], (float, int)))

    def test_var_uniform(self):
        dataset = self.generator.get_dataset()
        # check if the uniform dist var exists
        self.assertTrue("X2" in dataset.columns)
        # check if the first entry of the var is a float ot an int
        self.assertTrue(isinstance(dataset["X2"].iloc[0], (float, int)))
        # check if the var is in the desired range
        self.assertTrue(0 <= dataset["X2"].iloc[0] <= 1)

    def test_var_expression(self):
        dataset = self.generator.get_dataset()
        self.assertTrue("X3" in dataset.columns)
        # check if the expression has correctly combine the two variables
        for i in range(len(dataset)):
            self.assertAlmostEqual(
                dataset["X3"].iloc[i],
                dataset["X1"].iloc[i] + dataset["X2"].iloc[i],
            )

    def test_metadata(self):
        self.generator.add_var("X4", expression="X1*X2")
        self.generator.add_var(
            "X5", distribution="normal", dist_params={"loc": 0, "scale": 1}
        )
        metadata = self.generator.get_metadata()
        self.assertTrue("X4" in metadata)
        self.assertEqual(metadata["X4"]["expression"], "X1*X2")
        self.assertTrue("X5" in metadata)
        self.assertEqual(metadata["X5"]["distribution"], "normal")

    def test_remove_var(self):
        self.generator.add_var("X6", expression="2*X1+2*X2")
        self.generator.remove_var("X6")
        self.assertFalse("X6" in self.generator.get_dataset().columns)

    def test_save_csv(self):
        path = os.path.join("data", "synthetic", "test_dataset.csv")
        self.generator.add_var(
            "X7", distribution="normal", dist_params={"loc": 0, "scale": 1}
        )
        self.generator.save_as_csv()
        self.assertTrue(os.path.exists(path))
        os.remove(path)  # Cleanup

   


class TestExtractVar(unittest.TestCase):
    def test_extract_var_from_re(self):
        self.assertEqual(
            DatasetGenerator.extract_var_from_re("sin(x) + cos(y)"), ["x", "y"]
        )
        self.assertEqual(
            DatasetGenerator.extract_var_from_re("tan(a) * atan(b)"), ["a", "b"]
        )
        self.assertEqual(DatasetGenerator.extract_var_from_re("pi + e"), [])
        self.assertEqual(
            DatasetGenerator.extract_var_from_re("log(var1, var2)"), ["var1", "var2"]
        )
        self.assertEqual(
            DatasetGenerator.extract_var_from_re("z + sqrt(w) - pow(u, v)"),
            ["z", "w", "u", "v"],
        )
        self.assertEqual(
            DatasetGenerator.extract_var_from_re("radius * 2 * pi"), ["radius"]
        )


if __name__ == "__main__":
    # use this when runs as normal .py file
    # unittest.main()
    # use this in interactive mode juypiter Notebooks
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

# %%
