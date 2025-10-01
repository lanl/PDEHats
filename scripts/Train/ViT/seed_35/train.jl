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
    name_model = :ViT
    chs = 256
    epochs = 15
    ## Initial Train
    train(epochs, seed, name_model, chs)
    ## Continue Train (0 -> 1)
    ckpt_load = 0
    train(seed, name_model, chs, ckpt_load)
    ## Continue Train (1 -> 2)
    ckpt_load = 1
    train(seed, name_model, chs, ckpt_load)
    ## Continue Train (2 -> 3)
    ckpt_load = 2
    train(seed, name_model, chs, ckpt_load)
    ## Continue Train (3 -> 4)
    ckpt_load = 3
    train(seed, name_model, chs, ckpt_load)
    ## Continue Train (4 -> 5)
    ckpt_load = 4
    train(seed, name_model, chs, ckpt_load)
    ##
    return nothing
end
train()
