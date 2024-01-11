# TSP with a genetic algorithm

TSP is a problem of finding optimal path to visit all the given points/cities.

You can specify list of cities using `cities.json`:

```json
[
  [x0, y0],
  [x1, y1],
  ...
]
```

You can specify algorithm parameters using `config.json`:
```json
{
  "num_generations": 50,
  "population_size": 100,
  "crossover_rate": 0.7,
  "mutation_rate": 0.2
}
```

Individual is described as a list of indices representing specific cities in the list of available ones. \
Because the nature of TSP requires visiting all the cities, we cannot afford to lose any during mutations - for this
reason mutations are done by shuffling order of cities within an individual.

After running the GA, this program plots some data about the fitness function over generations. \
It also draws best and worst routes found, which is achieved by plotting city coordinates on a scatter plot, and then
drawing arrows based on the data of best and worst individuals.
