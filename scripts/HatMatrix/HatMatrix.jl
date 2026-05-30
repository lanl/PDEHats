##
using DrWatson
@quickactivate :PDEHats
##
using Lux, LuxCUDA
using Zygote, ComponentArrays
using ForwardDiff
using Random, Statistics
using MLUtils, NNlib
using LinearOperators, Krylov
using KrylovKit
using ConcreteStructs
using Optimisers
using LinearAlgebra
using Printf
using MLDataDevices
using Setfield
##
include(projectdir("scripts/HatMatrix/obs_batch.jl"))
include(projectdir("scripts/HatMatrix/diffs.jl"))
include(projectdir("scripts/HatMatrix/err_eqv.jl"))
include(projectdir("scripts/HatMatrix/lie.jl"))
include(projectdir("scripts/HatMatrix/bra_and_ket.jl"))
##
include(projectdir("scripts/HatMatrix/jacobian.jl"))
include(projectdir("scripts/HatMatrix/vjp.jl"))
##
include(projectdir("scripts/HatMatrix/qr.jl"))
include(projectdir("scripts/HatMatrix/qr_eqv.jl"))
include(projectdir("scripts/HatMatrix/eigen.jl"))
##
include(projectdir("scripts/HatMatrix/adam.jl"))
include(projectdir("scripts/HatMatrix/euclidean.jl"))
include(projectdir("scripts/HatMatrix/euclidean_corrected.jl"))
##
