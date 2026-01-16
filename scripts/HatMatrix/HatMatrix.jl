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
##
include(projectdir("scripts/HatMatrix/jacobian.jl"))
include(projectdir("scripts/HatMatrix/bra_and_ket.jl"))
include(projectdir("scripts/HatMatrix/chi_J_ket.jl"))
include(projectdir("scripts/HatMatrix/bra_J_chi_J_ket.jl"))
include(projectdir("scripts/HatMatrix/eigen.jl"))
include(projectdir("scripts/HatMatrix/obs_batch.jl"))
include(projectdir("scripts/HatMatrix/lie.jl"))
include(projectdir("scripts/HatMatrix/err_eqv.jl"))
include(projectdir("scripts/HatMatrix/diffs.jl"))
##
function save_hats()
    save_obs_batch()
    save_eigen()
    save_err_eqv()
    save_chi_g_J_ket()
    save_bra_J_g_chi_g_J_ket()
    save_diffs()
    return nothing
end
