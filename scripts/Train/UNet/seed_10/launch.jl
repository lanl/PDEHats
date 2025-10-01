# module load mpich
# MPIPreferences.use_jll_binary()
##
using DrWatson
@quickactivate :PDEHats
##
using MPI
## Script
seed = 10
name_model = :UNet
## MPI
path_script = projectdir("scripts/Train/$(name_model)/seed_$(seed)/train.jl")
n_gpu = 2
## RUN
run(`$(MPI.mpiexec()) -n $(n_gpu) julia --project=. -t auto $(path_script)`)
