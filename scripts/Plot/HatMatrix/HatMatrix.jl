##
using DrWatson
@quickactivate :PDEHats
##
using CairoMakie, MakiePublication, LaTeXStrings
using Statistics
using MLUtils
using LinearAlgebra
using Tullio
##
include(projectdir("scripts/Plot/HatMatrix/irreps.jl"))
include(projectdir("scripts/Plot/HatMatrix/plot_invariants.jl"))
include(projectdir("scripts/Plot/HatMatrix/plot_horizon.jl"))
include(projectdir("scripts/Plot/HatMatrix/plot_rkhs.jl"))
include(projectdir("scripts/Plot/HatMatrix/plot_compare.jl"))
include(projectdir("scripts/Plot/HatMatrix/plot_hists.jl"))
include(projectdir("scripts/Plot/HatMatrix/plot_t_series.jl"))
##
include(projectdir("scripts/HatMatrix/lie.jl"))
##
function plot_hat_matrix()
    plot_invariants()
    plot_horizon()
    plot_t_series()
    plot_compare()
    plot_hists()
    plot_rkhs()
    return nothing
end
##
function get_M_overlap(M)
    (T, N) = size(M)[1:2]
    M_overlap = map(Iterators.product(1:T, 1:N, 1:T, 1:N)) do (t1, n1, t2, n2)
        M_t1n1 = sqrt(-M[t1, n1, t1, n1] + 1.0f-6)
        M_t2n2 = sqrt(-M[t2, n2, t2, n2] + 1.0f-6)
        return M[t1, n1, t2, n2] / (M_t1n1 * M_t2n2 + 1.0f-6)
    end
    return M_overlap
end
