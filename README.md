## Implementation of CRO algorithm on a multilayered graph
Usage of Chemical Reaction Optimization Algorithm to optimize an objective function over a multilayered network. 
The algorithm is run on a sample case of a ventilator distribution supply chain for COVID-19 in Maharashtra.

### Variables
- _popSize_ = Initial number of molecules
- _buffer_ = Initial amount of energy stored in the surroundings
- _KELossRate_ = Minimum proportion of KE lost during a molecular collision
- _MoleColl_ = Controls chances of unimolecular or bimolecular reactions occuring
- _NODE_DEMAND_ = Nodewise demand of the final layer of the graph
- $\alpha$ = Number of ineffective collisions before which decomposition is triggered
- $\beta$ = Kinetic Energy threshold of a molecule below which synthesis is triggered

### Test Cases
The algorithm was used on COVID Data from three waves in Maharashtra. The goal was to minimize total delivery time of
ventilators from supplier in China to Hospital (Demand nodes) in various districts of Maharashtra. The three cases represent
different scenarios, ranging from to low to high demand.

### Usage
The main algorithm is written in rust. In order to run the algorithm, download the code and build the cargo package
from the top directory.
Other utility tools are written in python, found under the `utils` directory.
Sensitivity and Scenario analysis of the test cases are in the `analysis` directory, as a jupyter notebook.

##### Additional Note
Most of the constraints are hardcoded into the program, and code for each operator would need to be rewritten to run it on another optimization problem.
