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
    seed = 35
    name_data = :CE
    name_model = :ViT
    chs = 256
    ## From Initialization (Arm)
    # epochs = 100
    # train(name_data, epochs, seed, name_model, chs)
    ## From Checkpoint: Epoch 100 (Arm)
    # epoch_ckpt = 100
    # epochs = 30
    # train(seed, name_model, chs, name_data, epoch_ckpt, epochs)
    ## From Checkpoint: Epoch 130 (Arm)
    epoch_ckpt = 130
    epochs = 20
    train(seed, name_model, chs, name_data, epoch_ckpt, epochs)
    ##
    return nothing
end
train()
