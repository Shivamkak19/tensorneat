from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.common import ACT, AGG
from tensorneat.genome import BiasNode
from classification_problem import ClassificationProblem
from sb2 import StaticBackpropGenome
import jax
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt


def evaluate_population(algorithm, state, pop_nodes, pop_conns, problem):
    """Evaluate population with improved error handling and numerical stability"""
    fitness = []
    accuracies = []

    for i in range(len(pop_nodes)):
        try:
            # Forward pass with clipped predictions to prevent log(0)
            predictions = algorithm.genome.static_forward(
                state, (pop_nodes[i], pop_conns[i]), problem.train_data[0]
            )

            # print("DIAG PREDICTIONS:", predictions)

            # Add small epsilon and clip predictions to prevent numerical instability
            epsilon = 1e-7
            predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)

            # Binary cross entropy loss with improved numerical stability
            loss = -jnp.mean(
                problem.train_data[1] * jnp.log(predictions)
                + (1 - problem.train_data[1]) * jnp.log(1 - predictions)
            )

            # Calculate accuracy
            # print("data train diag:", problem.train_data[1])
            # print("prediction diag:", predictions)
            accuracy = jnp.mean((predictions > 0.5) == problem.train_data[1])
            # print("accuracy diag:", accuracy)

            # Use negative loss as fitness (higher is better)
            fitness.append(-loss)
            accuracies.append(float(accuracy))

        except Exception as e:
            print(f"Evaluation failed for individual {i}: {str(e)}")
            fitness.append(
                -float("inf")
            )  # Use -inf instead of NaN for failed evaluations
            accuracies.append(0.0)

    return jnp.array(fitness), jnp.array(accuracies)


def plot_decision_boundary(
    algorithm, state, params, problem, title="Decision Boundary"
):
    """Plot the decision boundary of the classifier"""
    # Create a mesh grid
    x_min, x_max = (
        problem.train_data[0][:, 0].min() - 0.5,
        problem.train_data[0][:, 0].max() + 0.5,
    )
    y_min, y_max = (
        problem.train_data[0][:, 1].min() - 0.5,
        problem.train_data[0][:, 1].max() + 0.5,
    )
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Make predictions
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = algorithm.genome.static_forward(state, params, jnp.array(mesh_points))
    Z = (Z > 0.5).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(
        problem.train_data[0][:, 0],
        problem.train_data[0][:, 1],
        c=problem.train_data[1],
        alpha=0.8,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"decision_boundary_{int(time.time())}.png")
    plt.close()


def main():
    # Initialize problem
    problem = ClassificationProblem(
        train_file="circle_train.csv", test_file="circle_test.csv"
    )

    # Create NEAT algorithm with improved parameters
    algorithm = NEAT(
        pop_size=50,
        species_size=5,
        survival_threshold=0.2,
        compatibility_threshold=3.0,
        genome=StaticBackpropGenome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=20,
            max_conns=40,
            learning_rate=0.01,
            training_steps=100,
            batch_size=32,
            weight_decay=0.0001,
            gradient_clip=1.0,
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.sigmoid,
        ),
    )

    # Create pipeline with improved parameters
    pipeline = Pipeline(
        algorithm=algorithm,
        problem=problem,
        generation_limit=50,  # More generations
        fitness_target=-0.1,
        seed=42,
    )

    # Initialize
    state = pipeline.setup()

    # Training tracking
    best_fitness = -float("inf")
    best_accuracy = 0.0
    best_params = None
    patience = 3
    no_improvement = 0

    # History tracking
    history = {
        "train_accuracy": [],
        "test_accuracy": [],
        "fitness": [],
        "generation_time": [],
    }

    # Training loop with improved monitoring
    for generation in range(pipeline.generation_limit):
        start_time = time.time()

        print(f"\nGeneration {generation}")

        # Get population
        pop_nodes, pop_conns = algorithm.ask(state)

        # Train population
        for i in range(len(pop_nodes)):
            try:
                # Train with progress tracking
                print(f"Training individual {i}/{len(pop_nodes)}", end="\r")
                trained_nodes, trained_conns = algorithm.genome.train(
                    state, pop_nodes[i], pop_conns[i], problem.train_data
                )
                pop_nodes = pop_nodes.at[i].set(trained_nodes)
                pop_conns = pop_conns.at[i].set(trained_conns)
            except Exception as e:
                print(f"\nTraining failed for individual {i}: {str(e)}")
                continue

        # Evaluate population
        fitness, accuracies = evaluate_population(
            algorithm, state, pop_nodes, pop_conns, problem
        )

        # print("FITNESS DIAG:", fitness)

        # Update state
        state = algorithm.tell(state, fitness)

        # Track best performance
        gen_best_idx = jnp.argmax(fitness)
        gen_best_fitness = fitness[gen_best_idx]
        gen_best_accuracy = accuracies[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_accuracy = gen_best_accuracy
            best_params = (pop_nodes[gen_best_idx], pop_conns[gen_best_idx])
            no_improvement = 0

            # Plot decision boundary for best model
            plot_decision_boundary(
                algorithm,
                state,
                best_params,
                problem,
                f"Decision Boundary - Generation {generation}",
            )
        else:
            no_improvement += 1

        # Calculate test metrics
        test_predictions = algorithm.genome.static_forward(
            state,
            (pop_nodes[gen_best_idx], pop_conns[gen_best_idx]),
            problem.test_data[0],
        )
        test_accuracy = jnp.mean((test_predictions > 0.5) == problem.test_data[1])

        # Update history
        generation_time = time.time() - start_time
        history["train_accuracy"].append(gen_best_accuracy)
        history["test_accuracy"].append(float(test_accuracy))
        history["fitness"].append(float(gen_best_fitness))
        history["generation_time"].append(generation_time)

        # Print progress
        print(f"\nGeneration {generation} completed in {generation_time:.2f}s")
        print(f"Best Fitness: {gen_best_fitness:.4f}")
        print(f"Train Accuracy: {gen_best_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"All-time Best Fitness: {best_fitness:.4f}")
        print(f"All-time Best Accuracy: {best_accuracy:.4f}")

        # Early stopping
        if no_improvement >= patience:
            print(
                "\nEarly stopping triggered - no improvement for",
                patience,
                "generations",
            )
            break

        # Check fitness target
        if gen_best_fitness >= pipeline.fitness_target:
            print("\nFitness target reached!")
            break

    print("\nTraining completed!")

    # Final evaluation
    if best_params is not None:
        print("\nFinal Evaluation:")
        # Train set performance
        train_predictions = algorithm.genome.static_forward(
            state, best_params, problem.train_data[0]
        )
        final_train_accuracy = jnp.mean(
            (train_predictions > 0.5) == problem.train_data[1]
        )
        print(f"Final Train Accuracy: {final_train_accuracy:.4f}")

        # Test set performance
        test_predictions = algorithm.genome.static_forward(
            state, best_params, problem.test_data[0]
        )
        final_test_accuracy = jnp.mean((test_predictions > 0.5) == problem.test_data[1])
        print(f"Final Test Accuracy: {final_test_accuracy:.4f}")

        # Plot final decision boundary
        plot_decision_boundary(
            algorithm, state, best_params, problem, "Final Decision Boundary"
        )

        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_accuracy"], label="Train")
        plt.plot(history["test_accuracy"], label="Test")
        plt.title("Accuracy vs Generation")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["fitness"])
        plt.title("Best Fitness vs Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.close()


if __name__ == "__main__":
    main()