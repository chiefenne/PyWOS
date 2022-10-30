
```diff
- WORK IN PROGRESS
```

# PyWOS
![](images/WOS_example.png)

Monte Carlo method based on a walk-on-spheres implementation for solving a 2D Poisson PDE written in Python.

Based on the excellent [video](https://youtu.be/bZbuKOxH71o) by Keenan Crane from Carnegie Mellon University and a code snippet he provided [here](https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/WoSPoisson2D.cpp.html).

The version presented here utilizes recursion.

Associated literature and [video](https://youtu.be/dXROl0KGPXc) from the original author:
["Grid-Free Monte Carlo for PDEs with Spatially Varying Coefficients"](https://arxiv.org/abs/2201.13240) by Sawhney, Seyb, Jarosz, Crane.

## Usage


Print help on usage:
```code
python Random_walks_Poisson_solver.py -h 
```
Do a single walk and plot it.
```code
python Random_walks_Poisson_solver.py -d -v 
```
Run the solver using seetings for number of walks, accuracy, and maximum number of steps per walk.
```code
python Random_walks_Poisson_solver.py -w 100 -e 0.01 -s 30 
```

## Requirements

- Python 3.9+ (due to the version of argparse which is used)
- Numpy
- Matplotlib
