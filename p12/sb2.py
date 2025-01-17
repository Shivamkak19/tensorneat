import jax
import jax.numpy as jnp
from functools import partial
import optax
from tensorneat.genome import DefaultGenome
from tensorneat.genome.utils import extract_gene_attrs, unflatten_conns


class StaticBackpropGenome(DefaultGenome):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_rate: float = 0.01,
        training_steps: int = 100,
        batch_size: int = 32,
        max_layers: int = 3,
        weight_decay: float = 0.0001,
        gradient_clip: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(num_inputs, num_outputs, *args, **kwargs)
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.max_layers = max_layers
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip

    @partial(jax.jit, static_argnums=(0,))
    def static_forward(self, state, params, inputs):
        nodes, conns = params

        # Extract node and connection attributes with NaN handling
        def safe_extract_attrs(gene, values):
            attrs = jax.vmap(extract_gene_attrs, in_axes=(None, 0))(gene, values)
            return jnp.nan_to_num(attrs, nan=0.0)  # Replace NaNs with zeros

        node_attrs = safe_extract_attrs(self.node_gene, nodes)
        conn_attrs = safe_extract_attrs(self.conn_gene, conns)

        # Create adjacency matrix with weights
        weights = jnp.zeros((self.max_nodes, self.max_nodes))
        for i in range(self.max_conns):
            in_idx = conns[i, 0]
            out_idx = conns[i, 1]
            weight = conn_attrs[i, 0]  # Assuming first attribute is weight

            # Convert indices to safe integers and clip to valid range
            in_idx = jnp.clip(in_idx, 0, self.max_nodes - 1).astype(int)
            out_idx = jnp.clip(out_idx, 0, self.max_nodes - 1).astype(int)

            # Replace this with better weight initialization
            weight = jax.random.uniform(
                jax.random.PRNGKey(42), (), minval=-1.0, maxval=1.0
            )
            weight = jnp.nan_to_num(weight, nan=0.0)  # Replace NaNs with 0

            weights = weights.at[in_idx, out_idx].set(weight)

        # Handle batch inputs
        batch_size = inputs.shape[0] if inputs.ndim > 1 else 1
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        # Initialize activations
        activations = jnp.zeros((batch_size, self.max_nodes))
        activations = activations.at[:, : self.num_inputs].set(inputs)

        # Process through layers
        for _ in range(self.max_layers):
            # Matrix multiplication for connections
            layer_output = jnp.matmul(activations, weights)

            # Add bias
            bias = node_attrs[:, 0]
            bias = jax.random.uniform(
                jax.random.PRNGKey(42), (), minval=-0.1, maxval=0.1
            )
            bias = jnp.nan_to_num(bias, nan=0.0)  # Replace NaNs with 0

            layer_output += bias

            # Apply ReLU activation (replace tanh if needed)
            layer_output = jnp.maximum(layer_output, 0)  # ReLU
            # For LeakyReLU:
            # layer_output = jnp.where(layer_output > 0, layer_output, 0.01 * layer_output)

            # Update activations
            activations = layer_output

        # Get output nodes
        outputs = activations[:, self.output_idx]

        # Apply output transform if specified
        if self.output_transform is not None:
            outputs = jax.vmap(self.output_transform)(outputs)
            outputs = jnp.clip(
                outputs, 1e-7, 1.0 - 1e-7
            )  # Clip for numerical stability

        return outputs.squeeze()

    def loss_fn(self, params, state, batch):
        """Loss function with L2 regularization and improved numerical stability"""
        inputs, targets = batch
        predictions = self.static_forward(state, params, inputs)

        # Binary cross entropy with numerical stability
        epsilon = 1e-7
        predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
        bce_loss = -jnp.mean(
            targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)
        )

        # L2 regularization on weights
        _, conns = params
        conn_attrs = jax.vmap(extract_gene_attrs, in_axes=(None, 0))(
            self.conn_gene, conns
        )
        l2_loss = self.weight_decay * jnp.sum(jnp.square(conn_attrs[:, 0]))

        return bce_loss + l2_loss

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, opt_state, params, batch):
        """Single training step with gradient clipping"""
        loss_val, grads = jax.value_and_grad(self.loss_fn)(params, state, batch)

        # Clip gradients
        grads = jax.tree_map(
            lambda g: jnp.clip(g, -self.gradient_clip, self.gradient_clip), grads
        )

        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss_val

    def train(self, state, nodes, conns, dataset):
        """Training loop with improved batch handling and learning rate schedule"""
        # Initialize optimizer with learning rate schedule
        schedule = optax.exponential_decay(
            init_value=self.learning_rate, transition_steps=100, decay_rate=0.95
        )

        self.optimizer = optax.chain(
            optax.clip(self.gradient_clip), optax.adam(learning_rate=schedule)
        )

        params = (nodes, conns)
        opt_state = self.optimizer.init(params)

        x_train, y_train = dataset
        data_size = len(x_train)

        # Training loop
        for step in range(self.training_steps):
            # Create batch
            idx = jax.random.randint(
                jax.random.PRNGKey(step), (self.batch_size,), 0, data_size
            )
            batch = (x_train[idx], y_train[idx])

            # Update parameters
            params, opt_state, _ = self.train_step(state, opt_state, params, batch)

        return params
