##
using DrWatson
@quickactivate :PDEHats
##
using Lux
using CairoMakie, MakiePublication, LaTeXStrings
using Statistics
using MLUtils
using LinearAlgebra
using Tullio
using Random
##
include(projectdir("scripts/Plot/HatMatrix/heatmap.jl"))
include(projectdir("scripts/Plot/HatMatrix/horizon.jl"))
include(projectdir("scripts/Plot/HatMatrix/diag.jl"))
include(projectdir("scripts/Plot/HatMatrix/hist.jl"))
include(projectdir("scripts/Plot/HatMatrix/rkhs.jl"))
include(projectdir("scripts/Plot/HatMatrix/utils.jl"))

include(projectdir("scripts/Plot/HatMatrix/eig.jl"))
include(projectdir("scripts/Plot/HatMatrix/dynamics.jl"))

include(projectdir("scripts/Plot/HatMatrix/Lie/Lie.jl"))
##
function plot_hat()
    plot_heatmap()
    plot_horizon()
    plot_diag()
    plot_hist()
    plot_rkhs()
    plot_eig()
    plot_dynamics()
    return nothing
end
function plot_hat_lie()
    plot_err_eqv_D4()
    plot_err_eqv_Z()
    plot_err_eqv_box()
    return nothing
end
