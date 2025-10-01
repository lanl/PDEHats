##
using DrWatson
@quickactivate :PDEHats
##
using Lux, LuxCUDA
using ComponentArrays
using Optimisers
using MLUtils, NNlib
using Random, Statistics
using FilePaths
using ConcreteStructs
##
using MPI: MPI
using MPIPreferences
using NCCL: NCCL
using Setfield
##
include(projectdir("scripts/Train/Train.jl"))
##
function train()
    ## Seed
    seed = 35
    ## Config
    name_model = :UNet
    chs = 24
    epochs = 30
    ## Initial Train
    train(epochs, seed, name_model, chs)
    ## Continue Train (0 -> 1)
    ckpt_load = 0
    train(seed, name_model, chs, ckpt_load)
    ## Continue Train (1 -> 2)
    ckpt_load = 1
    train(seed, name_model, chs, ckpt_load)
    ##
    return nothing
end
train()
