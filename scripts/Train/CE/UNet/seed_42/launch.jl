# module load mpich
# MPIPreferences.use_jll_binary()
##
using DrWatson
@quickactivate :PDEHats
##
using MPI
## Seed
seed = 42
## Model
name_model = :UNet
name_data = :CE
## MPI
path_script = projectdir(
    "scripts/Train/$(name_data)/$(name_model)/seed_$(seed)/train.jl"
)
n_gpu = 2
## RUN
run(`$(MPI.mpiexec()) -n $(n_gpu) julia $(path_script)`)
