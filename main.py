from functools import partial
from random import choices, randint, randrange, random
from typing import List, Tuple, Callable

import gym
from numpy import average

env = gym.make('MountainCar-v0')


Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


def generate_genome(Length: int) -> Genome:
    print("Genome generated")
    return choices([0, 1, 2], k=Length)


def generate_population(size: int, genome_length: int) -> Population:
    print("population generated")
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome) -> int:
    print("fitness started")
    observation = env.reset()
    env.render()
    for actions in genome:
        reward = env.step(actions)

    return reward[1]


def selection(population: Population, fitness_func: FitnessFunc) -> Population:
    print("started selection")
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=5
    )


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    print("crossover started")
    if len(a) != len(b):
        raise ValueError("Genomes must be of the same length")

    length = len(a)

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    print("started mutation")
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - randint(0, 2))
    return genome


def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc = fitness,
    selection_func: SelectionFunc = selection,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100
) -> Population:
    population = populate_func()


    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )

        if average(fitness_func(population[0])) <= -90:
            break

        next_generation = population[0:5]

    for j in range(int(len(population) / 2) - 4):
        parents = selection_func(population, fitness_func)
        offspring_a, offspring_b = crossover_func(parents[0], parents[1])
        offspring_a = mutation_func(offspring_a)
        offspring_b = mutation_func(offspring_b)
        next_generation += [offspring_a, offspring_b]

    population = next_generation


    return population, i


population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=50, genome_length=200
    )
)

print(f"number of generations: {generations}")
print(f"final genome : {population}")
