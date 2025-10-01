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
using ConcreteStructs
using Optimisers
using LinearAlgebra
using Interpolations
using Printf
##
include(projectdir("scripts/HatMatrix/bra_and_ket.jl"))
include(projectdir("scripts/HatMatrix/chi_J_ket.jl"))
include(projectdir("scripts/HatMatrix/bra_J_chi_J_ket.jl"))
include(projectdir("scripts/HatMatrix/jacobian.jl"))
include(projectdir("scripts/HatMatrix/krylov.jl"))
include(projectdir("scripts/HatMatrix/lie.jl"))
include(projectdir("scripts/HatMatrix/utils.jl"))
##
