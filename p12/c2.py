import wandb
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import uuid
from datetime import datetime
import os
from functools import partial
import argparse

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.common import ACT, AGG
from tensorneat.genome import BiasNode
from classification_problem import ClassificationProblem
from sb2 import StaticBackpropGenome

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--max-nodes", type=int, default=20, help="Max nodes")
    parser.add_argument("--max-conns", type=int, default=40, help="Max connections")
    parser.add_argument("--species-size", type=int, default=5, help="Number of species")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb-project", type=str, default="neat-backprop-classifier", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="shivamkak9-princeton-university", help="W&B entity/username")
    
    # Generate unique run name
    default_run_name = f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{str(uuid.uuid4())[:6]}"
    parser.add_argument("--wandb-name", type=str, default=default_run_name, help="W&B run name")
    
    return parser.parse_args()

def log_network_metrics(wandb_run, network_dict, generation):
    """Log network architecture metrics to W&B"""
    metrics = {
        'network/num_nodes': len(network_dict['nodes']),
        'network/num_connections': len(network_dict['conns']),
        'network/num_layers': len(network_dict['topo_layers'])
    }
    
    # Count activation functions
    act_funcs = {}
    for node_data in network_dict['nodes'].values():
        act = node_data['act']
        act_funcs[act] = act_funcs.get(act, 0) + 1
    
    for act, count in act_funcs.items():
        metrics[f'network/activation_{act}'] = count
        
    wandb_run.log(metrics, step=generation)

def visualize_and_log_network(genome, state, nodes, conns, generation, wandb_run):
    """Create and log network visualization"""
    try:
        network = genome.network_dict(state, nodes, conns)
        
        # Log network metrics
        log_network_metrics(wandb_run, network, generation)
        
        # Create visualization
        genome.visualize(
            network,
            save_path=f"temp_network_gen_{generation}.png",
            with_labels=True,
            figure_size=(12, 8),
            save_dpi=300  # Higher DPI for better quality
        )
        
        # Ensure the file was created
        if not os.path.exists(f"temp_network_gen_{generation}.png"):
            raise FileNotFoundError(f"Visualization file {viz_filename} was not created")
        
        # Log to wandb
        wandb_run.log({
            "network/graph": wandb.Image(f"temp_network_gen_{generation}.png")
        }, step=generation)
        
        # Cleanup
        # os.remove(viz_filename)
            
    except Exception as e:
        print(f"Warning: Failed to create/log network visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def train_with_logging(config):
    # Initialize wandb
    try:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_name,
            config=vars(config)
        )
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {str(e)}")
        run = None

    # Create problem instance
    problem = ClassificationProblem(
        train_file="circle_train.csv",
        test_file="circle_test.csv"
    )

    # Initialize algorithm
    algorithm = NEAT(
        pop_size=config.pop_size,
        species_size=config.species_size,
        survival_threshold=0.2,
        compatibility_threshold=3.0,
        genome=StaticBackpropGenome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=config.max_nodes,
            max_conns=config.max_conns,
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

    # Initialize pipeline
    pipeline = Pipeline(
        algorithm=algorithm,
        problem=problem,
        generation_limit=config.generations,
        fitness_target=-0.1,
        seed=config.seed
    )

    # Setup
    state = pipeline.setup()

    # Training loop
    best_fitness = float('-inf')
    best_genome = None
    patience = 5
    no_improvement = 0
    
    print("Starting training...")
    
    for generation in range(config.generations):
        start_time = time.time()
        
        # Get current population
        pop_nodes, pop_conns = algorithm.ask(state)
        
        # Train population
        for i in range(len(pop_nodes)):
            try:
                print(f"Training individual {i+1}/{len(pop_nodes)}", end='\r')
                trained_nodes, trained_conns = algorithm.genome.train(
                    state, pop_nodes[i], pop_conns[i], problem.train_data
                )
                pop_nodes = pop_nodes.at[i].set(trained_nodes)
                pop_conns = pop_conns.at[i].set(trained_conns)
            except Exception as e:
                print(f"\nWarning: Training failed for individual {i}: {str(e)}")
                continue
                
        print("\nEvaluating population...")
        
        # Evaluate population
        fitnesses = []
        accuracies = []
        for i in range(len(pop_nodes)):
            try:
                predictions = algorithm.genome.static_forward(
                    state, (pop_nodes[i], pop_conns[i]), problem.train_data[0]
                )
                
                # Calculate metrics
                epsilon = 1e-7
                predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
                loss = -jnp.mean(
                    problem.train_data[1] * jnp.log(predictions)
                    + (1 - problem.train_data[1]) * jnp.log(1 - predictions)
                )
                accuracy = jnp.mean((predictions > 0.5) == problem.train_data[1])
                
                fitnesses.append(-loss)  # Negative loss as fitness
                accuracies.append(float(accuracy))
            except Exception as e:
                print(f"Warning: Evaluation failed for individual {i}: {str(e)}")
                fitnesses.append(float('-inf'))
                accuracies.append(0.0)
        
        fitnesses = jnp.array(fitnesses)
        accuracies = jnp.array(accuracies)
        
        # Update algorithm state
        state = algorithm.tell(state, fitnesses)
        
        # Track best performance
        gen_best_idx = jnp.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_accuracy = accuracies[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = (pop_nodes[gen_best_idx], pop_conns[gen_best_idx])
            no_improvement = 0
        else:
            no_improvement += 1
            
        # Calculate test metrics
        if best_genome is not None:
            test_predictions = algorithm.genome.static_forward(
                state, best_genome, problem.test_data[0]
            )
            test_accuracy = float(jnp.mean((test_predictions > 0.5) == problem.test_data[1]))
        else:
            test_accuracy = 0.0
            
        # Log metrics
        generation_time = time.time() - start_time
        if run is not None:
            metrics = {
                'generation': generation,
                'train/accuracy_best': gen_best_accuracy,
                'train/accuracy_mean': float(accuracies.mean()),
                'train/fitness_best': float(gen_best_fitness),
                'train/fitness_mean': float(fitnesses.mean()),
                'test/accuracy': test_accuracy,
                'time/generation': generation_time
            }
            run.log(metrics, step=generation)
            
            # Visualize best network
            if best_genome is not None:
                visualize_and_log_network(
                    algorithm.genome,
                    state,
                    best_genome[0],
                    best_genome[1],
                    generation,
                    run
                )
        
        # Print progress
        print(f"Generation {generation} completed in {generation_time:.2f}s")
        print(f"Best Fitness: {gen_best_fitness:.4f}")
        print(f"Train Accuracy: {gen_best_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Early stopping
        if no_improvement >= patience:
            print(f"\nStopping early - no improvement for {patience} generations")
            break
            
        # Check fitness target
        if gen_best_fitness >= pipeline.fitness_target:
            print("\nFitness target reached!")
            break
            
    print("\nTraining completed!")
    
    if run is not None:
        run.finish()

def main():
    config = parse_args()
    train_with_logging(config)

if __name__ == "__main__":
    main()