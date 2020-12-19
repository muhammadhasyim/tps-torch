# tps-torch

A PyTorch-based package for string methods and ML-inspired path sampling methods. Before installing and building this project, it is important that you have CMake 2.8.12 above, any MPI implementation (OpenMPI or MPICH), and, of course, PyTorch (v 1.7.0 and above).

At the moment, building and installation is just:

```console

mkdir build; cd build
cmake ../
make -j6
pip install .

```

Listed files and folders:
- `tpstorch`, contains C++ and Python source files
- `1dbrownian`, contains scripts to run the 1D Quartic Potential study
- `muller-brown`, contains scripts to build and run the 2D Muller-Brown potential study with the FTS method. 
- `muller-brown-ml`, contains scripts to build and run the 2D Muller-Brown potential study with the PID BKE algorithm. 
