##
using DrWatson
@quickactivate :Archon
##
using CairoMakie, MakiePublication, LaTeXStrings
using Statistics
using LinearAlgebra
##
include(projectdir("scripts/Plot/HatMatrix/NeurReps/plot_dihedral.jl"))
include(projectdir("scripts/Plot/HatMatrix/NeurReps/plot_grid.jl"))
include(projectdir("scripts/Plot/HatMatrix/NeurReps/plot_lines.jl"))
include(projectdir("scripts/Plot/HatMatrix/NeurReps/plot_eqv_lines.jl"))
include(projectdir("scripts/Plot/HatMatrix/NeurReps/plot_eqv_dihedral.jl"))
include(projectdir("scripts/Plot/HatMatrix/NeurReps/plot_eqv_grid.jl"))
##
include(projectdir("scripts/HatMatrix/lie.jl"))
##
function plot_neurreps()
    plot_dihedral()
    # plot_grid()
    plot_lines()
    plot_eqv_lines()
    plot_eqv_dihedral()
    # plot_eqv_grid()
    return nothing
end
