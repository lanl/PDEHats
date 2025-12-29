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
    ## Seed
    seed = 10
    name_data = :NS
    name_model = :UNet
    chs = 24
    ## From Initialization (Arm)
    epochs = 100
    train(name_data, epochs, seed, name_model, chs)
    ##
    return nothing
end
train()
