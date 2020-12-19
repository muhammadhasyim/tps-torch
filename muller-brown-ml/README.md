This folder implements the PID-BKE method on a Muller-Brown potential.
To run, do
At the moment, building and installation is just:

```console

mkdir build; cd build
cmake ../
make -j6

```
Install it or move the created binary wherever you go.
The examples/ folder has some test cases, which can be run by entering the folder and
```console
mpirun -np 8 python run.py
```
