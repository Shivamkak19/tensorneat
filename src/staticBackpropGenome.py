import jax
import jax.numpy as jnp
from functools import partial
import optax
from tensorneat.genome import DefaultGenome
from tensorneat.genome.utils import extract_gene_attrs

class StaticBackpropGenome(DefaultGenome):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_rate: float = 0.01,
        training_steps: int = 100,
        batch_size: int = 32,
        weight_decay: float = 0.0001,
        gradient_clip: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(num_inputs, num_outputs, *args, **kwargs)
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip

    @partial(jax.jit, static_argnums=(0,))
    def static_forward(self, state, params, inputs):
        """JAX-compatible forward pass implementation"""
        nodes, conns = params
        
        # Ensure inputs are 2D
        inputs = jnp.atleast_2d(inputs)
        batch_size = inputs.shape[0]
        
        # Extract node and connection attributes
        node_attrs = extract_gene_attrs(self.node_gene, nodes)
        conn_attrs = extract_gene_attrs(self.conn_gene, conns)
        
        # Initialize activation storage
        activations = jnp.zeros((batch_size, self.max_nodes))
        
        # Set input activations
        activations = activations.at[:, :self.num_inputs].set(inputs)
        
        # Create adjacency matrix with weights
        weights = jnp.zeros((self.max_nodes, self.max_nodes))
        valid_conns = ~jnp.any(jnp.isnan(conns), axis=1)
        
        def update_weights(i, w):
            return jax.lax.cond(
                valid_conns[i],
                lambda w: w.at[int(conns[i, 0]), int(conns[i, 1])].set(conn_attrs[i, 0]),
                lambda w: w,
                w
            )
        
        weights = jax.lax.fori_loop(0, len(conns), update_weights, weights)
        
        # Process nodes in topological order
        def process_node(i, acts):
            node_valid = ~jnp.any(jnp.isnan(nodes[i]))
            
            def compute_activation():
                # Get incoming connections
                incoming = weights[:, i]
                
                # Compute weighted sum
                weighted_sum = jnp.dot(acts, incoming)
                
                # Add bias and apply activation
                pre_activation = weighted_sum + node_attrs[i, 0]  # Add bias
                
                # Different activation for output nodes
                is_output = jnp.any(self.output_idx == i)
                
                return jax.lax.cond(
                    is_output,
                    lambda x: x,  # No activation for output nodes
                    lambda x: jnp.tanh(x),  # tanh activation for hidden nodes
                    pre_activation
                )
            
            # Only update if node is valid and not an input node
            is_input = jnp.any(self.input_idx == i)
            should_update = node_valid & ~is_input
            
            new_activation = jax.lax.cond(
                should_update,
                lambda: compute_activation(),
                lambda: acts[:, i]
            )
            
            return acts.at[:, i].set(new_activation)
        
        # Process all nodes
        activations = jax.lax.fori_loop(0, self.max_nodes, process_node, activations)
        
        # Get output activations
        outputs = activations[:, self.output_idx]
        
        # Apply output transform if specified
        if self.output_transform is not None:
            outputs = self.output_transform(outputs)
        
        return jnp.squeeze(outputs)

    def loss_fn(self, params, state, batch):
        """Calculate loss with numerical stability"""
        inputs, targets = batch
        predictions = self.static_forward(state, params, inputs)
        
        # Binary cross entropy with improved numerical stability
        epsilon = 1e-7
        predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
        bce_loss = -jnp.mean(
            targets * jnp.log(predictions) + 
            (1 - targets) * jnp.log(1 - predictions)
        )
        
        # L2 regularization
        _, conns = params
        valid_weights = jnp.where(
            jnp.isnan(conns[:, -1]), 
            0.0, 
            conns[:, -1]
        )
        l2_loss = self.weight_decay * jnp.mean(jnp.square(valid_weights))
        
        return bce_loss + l2_loss

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, opt_state, params, batch):
        """Single training step with gradient clipping"""
        loss_val, grads = jax.value_and_grad(self.loss_fn)(params, state, batch)
        
        # Clip gradients
        grads = jax.tree_map(
            lambda g: jnp.clip(g, -self.gradient_clip, self.gradient_clip),
            grads
        )
        
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss_val

    def train(self, state, nodes, conns, dataset):
        """Training loop with learning rate schedule"""
        # Initialize optimizer
        schedule = optax.exponential_decay(
            init_value=self.learning_rate,
            transition_steps=50,
            decay_rate=0.95
        )
        
        self.optimizer = optax.chain(
            optax.clip(self.gradient_clip),
            optax.adam(learning_rate=schedule)
        )
        
        params = (nodes, conns)
        opt_state = self.optimizer.init(params)
        
        x_train, y_train = dataset
        data_size = len(x_train)
        
        # Training loop with improved batch sampling
        def train_loop(step, carry):
            params, opt_state = carry
            
            # Create batch indices
            key = jax.random.PRNGKey(step)
            idx = jax.random.randint(key, (self.batch_size,), 0, data_size)
            batch = (x_train[idx], y_train[idx])
            
            # Update parameters
            params, opt_state, _ = self.train_step(state, opt_state, params, batch)
            
            return params, opt_state
        
        # Run training loop
        final_params, _ = jax.lax.fori_loop(
            0, 
            self.training_steps, 
            train_loop, 
            (params, opt_state)
        )
        
        return final_params