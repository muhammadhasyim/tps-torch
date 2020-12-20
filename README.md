# tps-torch

A PyTorch-based package for string methods and ML-inspired path sampling methods, as part of EECS 281A final project. In particular, we've implemented two path-finding algorithms:
- The Finite-Temperature String (FTS) Method, as outlined in Vanden-Eijnden, E., & Venturoli, M. (2009). "Revisiting the finite temperature string method for the calculation of reaction tubes and free energies". *The Journal of chemical physics, 130(19), 05B605*. 
- PID Control for solving Backward Kolmogorov Equation (BKE), an algorithm we constructed based on a deep learning approach outlined in Rotskoff, Grant M., and Eric Vanden-Eijnden. "Learning with rare data: Using active importance sampling to optimize objectives dominated by rare events." *arXiv preprint arXiv:2008.06334 (2020)*.

Before installing and building this project, it is important that you have CMake 2.8.12 above, any MPI implementation (OpenMPI or MPICH), and, of course, PyTorch (v 1.7.0 and above). 

At the moment, building and installation is just:

```console

mkdir build; cd build
cmake ../
make -j6
pip install .

```

Note that the code only supports runs on CPUs. 

Listed files and folders:
- `tpstorch`, contains C++ and Python source files
- `1dbrownian`, contains scripts to run the 1D Quartic Potential study
- `muller-brown`, contains scripts to build and run the 2D Muller-Brown potential study with the FTS method. 
- `muller-brown-ml`, contains scripts to build and run the 2D Muller-Brown potential study with the PID BKE algorithm. 
