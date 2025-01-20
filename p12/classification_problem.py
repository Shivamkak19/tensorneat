import jax
import jax.numpy as jnp
from tensorneat.problem.func_fit import FuncFit
import pandas as pd


class ClassificationProblem(FuncFit):
    def __init__(self, train_file: str, test_file: str):
        super().__init__(error_method="mse")

        # Load and normalize data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Normalize features
        train_features = train_df[["x", "y"]].values
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0)

        train_features = (train_features - mean) / (std + 1e-8)
        test_features = (test_df[["x", "y"]].values - mean) / (std + 1e-8)

        # Convert to jax arrays
        self.train_data = (
            jnp.array(train_features, dtype=jnp.float32),
            jnp.array(train_df["label"].values, dtype=jnp.float32)[:, None],
        )

        self.test_data = (
            jnp.array(test_features, dtype=jnp.float32),
            jnp.array(test_df["label"].values, dtype=jnp.float32)[:, None],
        )

        self._input_shape = self.train_data[0].shape
        self._output_shape = self.train_data[1].shape

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape