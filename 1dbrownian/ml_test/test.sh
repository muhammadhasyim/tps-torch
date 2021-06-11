#!/bin/bash

mpirun -n 20 python run_vanilla.py
mpirun -n 20 python run_fts.py
mpirun -n 20 python run_fts_cl.py
mpirun -n 20 python run_cl.py
