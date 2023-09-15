from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.func_fit import XOR, FuncFitConfig

if __name__ == '__main__':
    # running config
    config = Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=-1e-2,
            pop_size=10000
        ),
        neat=NeatConfig(
            inputs=2,
            outputs=1
        ),
        gene=NormalGeneConfig(),
        problem=FuncFitConfig(
            error_method='rmse'
        )
    )
    # define algorithm: NEAT with NormalGene
    algorithm = NEAT(config, NormalGene)
    # full pipeline
    pipeline = Pipeline(config, algorithm, XOR)
    # initialize state
    state = pipeline.setup()
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)