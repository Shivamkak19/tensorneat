from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.common import ACT, AGG
from tensorneat.genome import BiasNode
from classification_problem import ClassificationProblem
from backpropGenome import BackpropGenome
import jax.numpy as jnp
from backpropGenome import BackpropGenome


def evaluate_population(algorithm, state, pop_nodes, pop_conns, problem):
    """Evaluate all individuals in population"""
    fitness = []

    for i in range(len(pop_nodes)):
        try:
            # Forward pass for predictions
            predictions = algorithm.genome.static_forward(
                state, (pop_nodes[i], pop_conns[i]), problem.train_data[0]
            )

            # Calculate loss
            epsilon = 1e-7
            predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
            loss = -jnp.mean(
                problem.train_data[1] * jnp.log(predictions)
                + (1 - problem.train_data[1]) * jnp.log(1 - predictions)
            )
            fitness.append(-loss)  # Negative loss is fitness
        except Exception as e:
            print(f"Evaluation failed for individual {i}: {str(e)}")
            fitness.append(-float("inf"))

    return jnp.array(fitness)


def main():
    # Initialize the classification problem
    problem = ClassificationProblem(
        train_file="circle_train.csv", test_file="circle_test.csv"
    )

    # Create NEAT algorithm with static backprop genome
    algorithm = NEAT(
        pop_size=50,
        species_size=5,
        survival_threshold=0.2,
        compatibility_threshold=3.0,
        genome=BackpropGenome(
            num_inputs=2,  # x, y coordinates
            num_outputs=1,  # binary classification
            max_nodes=20,
            max_conns=40,
            max_layers=3,  # Maximum number of hidden layers
            learning_rate=0.01,
            training_steps=10,
            batch_size=32,
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.sigmoid,
        ),
    )

    # Create pipeline
    pipeline = Pipeline(
        algorithm=algorithm,
        problem=problem,
        generation_limit=2,
        fitness_target=-0.1,
        seed=42,
    )

    # Initialize state
    state = pipeline.setup()

    # Training loop
    # Training loop
    for generation in range(pipeline.generation_limit):
        print("EPOCH:", generation)

        # Get population
        pop_nodes, pop_conns = algorithm.ask(state)

        # Train each individual
        for i in range(len(pop_nodes)):
            try:
                trained_nodes, trained_conns = algorithm.genome.train(
                    state, pop_nodes[i], pop_conns[i], problem.train_data
                )
                pop_nodes = pop_nodes.at[i].set(trained_nodes)
                pop_conns = pop_conns.at[i].set(trained_conns)
                print(f"Individual {i} trained successfully")
            except Exception as e:
                print(f"Training failed for individual {i}: {str(e)}")
                continue

        # Evaluate population
        fitness = evaluate_population(algorithm, state, pop_nodes, pop_conns, problem)

        # Update algorithm state
        state = algorithm.tell(state, fitness)

        # Print progress
        try:
            best_idx = jnp.argmax(fitness)
            predictions = algorithm.genome.static_forward(
                state, (pop_nodes[best_idx], pop_conns[best_idx]), problem.test_data[0]
            )
            test_accuracy = jnp.mean((predictions > 0.5) == problem.test_data[1])
            print(
                f"Generation {generation}: "
                f"Best Fitness = {fitness[best_idx]:.4f}, "
                f"Test Accuracy = {test_accuracy:.4f}"
            )
        except Exception as e:
            print(f"Error evaluating best individual: {str(e)}")

        if jnp.max(fitness) >= pipeline.fitness_target:
            print("Fitness target reached!")
            break

    print("Training completed!")


if __name__ == "__main__":
    main()
