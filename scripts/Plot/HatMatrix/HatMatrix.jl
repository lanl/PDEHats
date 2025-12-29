##
using DrWatson
@quickactivate :PDEHats
##
using CairoMakie, MakiePublication, LaTeXStrings
using Statistics
using MLUtils
using LinearAlgebra
# using Tullio
##
# include(projectdir("scripts/Plot/HatMatrix/plot_horizon.jl"))
# include(projectdir("scripts/Plot/HatMatrix/plot_rkhs.jl"))
# include(projectdir("scripts/Plot/HatMatrix/plot_compare.jl"))
# include(projectdir("scripts/Plot/HatMatrix/plot_hists.jl"))
# include(projectdir("scripts/Plot/HatMatrix/plot_t_series.jl"))
include(projectdir("scripts/Plot/HatMatrix/heatmap.jl"))
include(projectdir("scripts/Plot/HatMatrix/rkhs.jl"))
include(projectdir("scripts/Plot/HatMatrix/horizon.jl"))
include(projectdir("scripts/Plot/HatMatrix/time_series.jl"))
include(projectdir("scripts/Plot/HatMatrix/eig.jl"))
include(projectdir("scripts/Plot/HatMatrix/Lie/Lie.jl"))
##
function plot_hat()
    plot_heatmap()
    plot_horizon()
    plot_time_series()
    # plot_rkhs()
    return nothing
end
function plot_lie()
    plot_err_eqv_Z()
    plot_err_eqv_D4()
    plot_influence_Z()
    plot_influence_D4()
    return nothing
end
##
function get_hats(
    name_model::Symbol,
    name_data::Symbol,
    bra_fn::Symbol,
    bra_g::Symbol,
    loss_fn::Symbol,
)
    ##
    ket_fn = :ket_C_smse
    ket_g = :g_identity
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    N_Obs = length(seeds) * length(idx_NTs)
    ##
    dir_model = projectdir("results/HatMatrix/$(name_data)/$(name_model)")
    name_file_load = "chi_$(ket_g)_J_$(ket_fn)"
    name_file_save = "$(bra_fn)_J_$(bra_g)_" * name_file_load
    result_paths = PDEHats.find_files(dir_model, name_file_save, ".jld2")
    @assert length(result_paths) == N_Obs
    err_paths = PDEHats.find_files_by_suffix(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv_$(bra_g).jld2",
    )
    @assert length(err_paths) == N_Obs
    hats_array = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        idx_rp = idx_NT.idx_rp
        idx_crp = idx_NT.idx_crp
        idx_rpui = idx_NT.idx_rpui
        dir_batch = savename(
            (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
        )
        path_hat = only(
            filter(
                r -> occursin("seed_$(seed)", r) && occursin(dir_batch, r),
                result_paths,
            ),
        )
        path_err = only(
            filter(
                r -> occursin("seed_$(seed)", r) && occursin(dir_batch, r),
                err_paths,
            ),
        )
        hat = load(path_hat)["bra_J_chi_J_ket"]
        errs = load(path_err)["err_eqv"]
        (T, N) = size(errs)
        errs_r = reshape(errs, T, N, 1, 1)
        hat_normed = hat ./ errs_r
        return hat_normed
    end
    return vec(hats_array)
end
##
