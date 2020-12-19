## fts_test

-   `brownian_fts.py` defines the overdamped Langevin dynamics simulator by a class `BrownianParticle` and an optimizer class that updates the path `CustomFTSMethod`
-   `run.py` runs the simulation. 
-   `animate.py` provides animation of each replica of the dynamics, to visualize what's happening. 
-   `generate_validate.py` generate data for the validation method, by taking in the data representing the transition state ensemble (TSE). 
-   `validate.py` performs the validation method. It also generates the plots that shows how committor is computed in the FTS method. 

To run the simulation with MPI and 11 processes, do:

```console
mpirun -n 11 python run.py
```

Odd numbers is desirable because then the TSE ensemble is just the middle process. To improve performance, add a `--bind-to core` flag. 

To generate validation dataset, run with MPI with as many number processes needed. The number of data available multiplies

```console
mpirun -n 10 python generate_validate.py
```
