##
using DrWatson
@quickactivate :PDEHats
##
using CairoMakie, MakiePublication, LaTeXStrings
using Statistics
using MLUtils
using LinearAlgebra
using Tullio
using DataInterpolations, RegularizationTools
using Random
##
include(projectdir("scripts/Plot/HatMatrix/heatmap.jl"))
include(projectdir("scripts/Plot/HatMatrix/rkhs.jl"))
include(projectdir("scripts/Plot/HatMatrix/horizon.jl"))
include(projectdir("scripts/Plot/HatMatrix/diag.jl"))
include(projectdir("scripts/Plot/HatMatrix/eig.jl"))
include(projectdir("scripts/Plot/HatMatrix/utils.jl"))
include(projectdir("scripts/Plot/HatMatrix/Lie/Lie.jl"))
##
function plot_hat()
    plot_heatmap()
    plot_horizon()
    plot_rkhs()
    plot_diag()
    plot_eig()
    return nothing
end
function plot_lie()
    plot_err_eqv_D4()
    plot_influence_D4()
    plot_err_eqv_Z()
    plot_influence_Z()
    plot_err_eqv_box()
    plot_influence_box()
    return nothing
end
##
