## fts_test

-   `brownian_fts.py` defines the overdamped Langevin dynamics simulator by a class `BrownianParticle` and an optimizer class that updates the path `CustomFTSMethod`
-   `run.py` runs the simulation. 
-   `animate.py` provides animation of each replica of the dynamics, to visualize what's happening. 
-   `validate.py` performs the validation method, by taking in the data representing the transition state ensemble (TSE). It also generates the plots that shows how committor is computed in the FTS method. 

To run the files with MPI and 6 processes, do:

```console
mpirun -n 6 python run.py
```
To improve performance, add a `--bind-to core` flag. 
