import json
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from deap import creator, base, tools, algorithms
from deap.base import Toolbox
from deap.tools import Logbook

CONFIG_FILE_NAME = "config.json"
CITIES_FILE_NAME = "cities.json"


def load_cities() -> List[Tuple[int, int]]:
    with (open(CITIES_FILE_NAME, "r")) as cities_file:
        return json.load(cities_file)


def load_config() -> Dict[str, float]:
    with open(CONFIG_FILE_NAME, "r") as config_file:
        return json.load(config_file)


def init_deap():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)


def evaluate(_ind: List[int], _cities: List[Tuple[int, int]]) -> Tuple[int]:
    return sum(
        np.linalg.norm(
            np.array(_cities[_ind[i]]) - np.array(_cities[_ind[i - 1]])
        ) for i in range(len(_cities))
    ),


def init_toolbox(_cities: List[Tuple[int, int]], _mutation_rate: float) -> Toolbox:
    _num_cities = len(_cities)
    _toolbox = base.Toolbox()
    _toolbox.register("indices", random.sample, range(_num_cities), _num_cities)
    _toolbox.register("individual", tools.initIterate, creator.Individual, _toolbox.indices)
    _toolbox.register("population", tools.initRepeat, list, _toolbox.individual)
    _toolbox.register("evaluate", lambda ind: evaluate(ind, _cities))
    _toolbox.register("mate", tools.cxPartialyMatched)
    _toolbox.register("mutate", tools.mutShuffleIndexes, indpb=_mutation_rate)
    _toolbox.register("select", tools.selRoulette)
    return _toolbox


def plot_stats(_stats: Logbook):
    gen = [s["gen"] for s in _stats]
    avg = [s["avg"] for s in _stats]
    std = [s["std"] for s in _stats]
    min_fit = [s["min"] for s in _stats]
    max_fit = [s["max"] for s in _stats]

    plt.figure(figsize=(20, 10))
    plt.plot(gen, avg, label="Average fitness")
    plt.plot(gen, min_fit, label="Minimum fitness")
    plt.plot(gen, max_fit, label="Maximum fitness")
    plt.fill_between(
        gen,
        [avg[i] - std[i] for i in range(len(_stats))],
        [avg[i] + std[i] for i in range(len(_stats))],
        color="gray", alpha=.2, label="Std deviation"
    )
    plt.title("Population fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()


def plot_route(_cities: List[Tuple[int, int]], _route: List[int], _title: str):
    _route_x = [_cities[i][0] for i in _route]
    _route_y = [_cities[i][1] for i in _route]

    plt.figure(figsize=(12, 12))
    plt.scatter(_route_x[1:-1], _route_y[1:-1], c="black", marker="o")
    plt.scatter(_route_x[0], _route_y[0], color="green", label="Start")
    plt.scatter(_route_x[-1], _route_y[-1], color="red", label="End")
    plt.xlim(0, len(_cities) + 1)
    plt.xticks(range(len(_cities) + 1))
    plt.ylim(0, len(_cities) + 1)
    plt.yticks(range(len(_cities) + 1))
    for i in range(len(_route) - 1):
        plt.arrow(
            _route_x[i],
            _route_y[i],
            _route_x[i + 1] - _route_x[i],
            _route_y[i + 1] - _route_y[i],
            shape="full",
            lw=0.5,
            length_includes_head=True,
            head_width=0.1,
            color="black"
        )
    plt.title(_title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    cities = load_cities()
    config = load_config()
    num_generations, population_size, crossover_rate, mutation_rate = config.values()
    init_deap()
    toolbox = init_toolbox(cities, mutation_rate)
    population = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    _, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=crossover_rate,
        mutpb=mutation_rate,
        ngen=num_generations,
        stats=stats,
        verbose=True
    )
    plot_stats(logbook)

    best_route = tools.selBest(population, k=1)[0]
    worst_route = tools.selWorst(population, k=1)[0]
    plot_route(cities, best_route, "Best route")
    plot_route(cities, worst_route, "Worst route")
