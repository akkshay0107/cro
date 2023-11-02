## Implementation of CRO algorithm on a multilayered graph
Usage of Chemical Reaction Optimization Algorithm (and Particle Swarm Optimization) to optimize an objective function over a multilayered network. 
The algorithm is run on sample cases of a ventilator distribution supply chain for COVID-19.

### Variables
- _popSize_ = Initial number of molecules
- _buffer_ = Initial amount of energy stored in the surroundings
- _KELossRate_ = Minimum proportion of KE lost during a molecular collision
- _MoleColl_ = Controls chances of unimolecular or bimolecular reactions occuring
- _NODE_DEMAND_ = Nodewise demand of the final layer of the graph
- $\alpha$ = Number of ineffective collisions before which decomposition is triggered
- $\beta$ = Kinetic Energy threshold of a molecule below which synthesis is triggered

### Branches
- ```master``` = Ordinary CRO algorithm
- ```pso``` = PSO algorithm (to compare with the performance of CRO)
- ```priority``` = CRO algorithm with priority. Distributes demand greedily to larger vehicles.

### Test Cases

Case 1: Maharashtra Data

The algorithm was used on COVID Data from three waves in Maharashtra. The goal was to minimize total delivery time of
ventilators from supplier in China to Hospital (Demand nodes) in various districts of Maharashtra. The three cases represent different scenarios, ranging from to low to high demand. In this case, there is only one supplier, and only one type of vehicle to be chosen at a given point (algorithm doesn't need to distribute demand among different types of vehicles).

```master``` and ```pso``` branch are coded to solve this case.

Case 2 : Pan India Data

This algorithm was used on COVID Data from three waves over India. The goal was to optimize the total delivery time once again. This case has increased complexity as compared to the previous one. There are more than 1 suppliers, and choice of 3 types of vehicles at a given point. In this case, the algorithm has to distribute demand among the three types of vehicles as well.

```priority``` branch is coded to solve this case. (Last 3 logs of ```master``` branch contain results of ordinary cro on case 2)

### Usage
The main algorithm is written in rust. In order to run the algorithm, clone the repo and build the cargo package
from the top directory.
Other utility tools are written in python, found under the `utils` directory.

##### Additional Note
Most of the constraints are hardcoded into the program, and code for each operator would need to be rewritten to run it on another optimization problem.
