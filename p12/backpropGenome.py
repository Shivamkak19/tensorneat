from tensorneat.genome import DefaultGenome
from tensorneat.genome.utils import extract_gene_attrs, set_gene_attrs
import jax
import jax.numpy as jnp
from functools import partial
import optax


class BackpropGenome(DefaultGenome):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_rate: float = 0.01,
        training_steps: int = 100,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(num_inputs, num_outputs, *args, **kwargs)
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.batch_size = batch_size

    @partial(jax.jit, static_argnums=(0,))
    def compute_loss(self, state, nodes, conns, batch):
        """Compute loss for a batch of data"""
        inputs, targets = batch

        # Transform the network
        transformed = self.transform(state, nodes, conns)

        # Get predictions
        predictions = jax.vmap(self.forward, in_axes=(None, None, 0))(
            state, transformed, inputs
        )

        # Calculate binary cross entropy loss
        epsilon = 1e-7
        predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
        loss = -jnp.mean(
            targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)
        )

        return loss

    @partial(jax.jit, static_argnums=(0,))
    def update_step(self, state, opt_state, params, batch):
        """Single optimization step"""
        nodes, conns = params

        # Compute gradients
        loss_fn = lambda p: self.compute_loss(state, p[0], p[1], batch)
        grads = jax.grad(loss_fn)((nodes, conns))

        # Update using optax
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates((nodes, conns), updates)

        return new_opt_state, new_params, None  # None is for carry

    def train(self, state, nodes, conns, dataset):
        """Train the network using backpropagation"""
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = self.optimizer.init((nodes, conns))

        # Get data
        x_train, y_train = dataset
        num_samples = len(x_train)

        # Create static training loop using scan
        def train_epoch(carry, _):
            opt_state, params = carry

            # Create batch indices
            idx = jax.random.permutation(jax.random.PRNGKey(0), num_samples)[
                : self.batch_size
            ]

            # Create batch
            batch = (x_train[idx], y_train[idx])

            # Update parameters
            new_opt_state, new_params, _ = self.update_step(
                state, opt_state, params, batch
            )

            return (new_opt_state, new_params), None

        # Run training loop with scan
        init_carry = (opt_state, (nodes, conns))
        (final_opt_state, (final_nodes, final_conns)), _ = jax.lax.scan(
            train_epoch,
            init_carry,
            None,  # No array to scan over
            length=self.training_steps,
        )

        return final_nodes, final_conns

    def get_params(self, state, nodes, conns):
        """Extract trainable parameters"""
        return nodes, conns

    def set_params(self, state, nodes, conns, params):
        """Set trainable parameters"""
        return params
