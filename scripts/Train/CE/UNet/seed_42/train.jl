##
using DrWatson
@quickactivate :PDEHats
##
using Lux, LuxCUDA
using ComponentArrays
using Optimisers
using MLUtils, NNlib
using Random, Statistics
using ConcreteStructs
using MLDataDevices
using ADTypes
using Setfield
##
using MPI: MPI
using MPIPreferences
using NCCL: NCCL
##
include(projectdir("scripts/Train/Train.jl"))
##
function train()
    ## Setup
    seed = 42
    name_data = :CE
    name_model = :UNet
    chs = 24
    ## From Initialization (Arm)
    epochs = 150
    train(name_data, epochs, seed, name_model, chs)
    ##
    return nothing
end
train()
