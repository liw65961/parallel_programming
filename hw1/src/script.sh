#!/bin/bash
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -c 2
#SBATCH -J hw1


module load mpi
make
srun -N 2 -n 8 ./hw1 ../testcases/01.in ../output/01.out

