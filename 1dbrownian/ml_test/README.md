## ml_test

-   `brownian_ml.py` defines the overdamped Langevin dynamics simulator by a class `BrownianParticle`, loss function `BrownianLoss`, and the neural network `CommittorNet` 
-   `run.py` runs the simulation and also the generation of validation dataset
-   `plot.py` provides plot for tracking positions, loss function as a function of iterations, etc.

To run the simulation with MPI and 11 processes, do:

```console
mpirun -n 11 python run.py
```

Odd numbers is desirable because then the TSE ensemble is just the middle process. To improve performance, add a `--bind-to core` flag. 
