##
function get_hat_normed(
    name_model::Symbol,
    name_data::Symbol,
    bra_fn::Symbol,
    bra_g::Symbol,
    loss_fn::Symbol,
    c_inds::Vector{<:CartesianIndex},
)
    ##
    seeds = (10, 35, 42)
    if (name_data == :NS) && (name_model == :ViT)
        seeds = (10, 42)
    end
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    ##
    hats_seed_batch = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        hat_normed = get_hat_normed(
            name_model, name_data, bra_fn, bra_g, loss_fn, seed, idx_NT
        )
        hats = map(c -> hat_normed[c], c_inds)
        return mean(hats)
    end
    ##
    return hats_seed_batch
end
##
function get_hat_normed(
    name_model::Symbol,
    name_data::Symbol,
    bra_fn::Symbol,
    bra_g::Symbol,
    loss_fn::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int},
)
    ##
    hat = get_hat(name_model, name_data, bra_fn, bra_g, loss_fn, seed, idx_NT)
    (T, N) = size(hat)[1:2]
    hat_r = reshape(hat, (T * N, T * N))
    hat_r_std = std(hat_r; dims=2)
    hat_r_normed = hat_r ./ hat_r_std
    hat_normed = reshape(hat_r_normed, size(hat))
    return hat_normed
end
##
function get_hat(
    name_model::Symbol,
    name_data::Symbol,
    bra_fn::Symbol,
    bra_g::Symbol,
    loss_fn::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int},
)
    ##
    ket_fn = :ket_C_smse
    ket_g = :g_identity
    ##
    dir_model = projectdir("results/HatMatrix/$(name_data)/$(name_model)")
    name_file_load = "chi_$(ket_g)_J_$(ket_fn)"
    name_file_save = "$(bra_fn)_J_$(bra_g)_" * name_file_load
    result_paths = PDEHats.find_files(dir_model, name_file_save, ".jld2")
    ##
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
    hat = load(path_hat)["bra_J_chi_J_ket"]
    return Float64.(hat)
end
##
function get_err(
    name_model::Symbol,
    name_data::Symbol,
    bra_fn::Symbol,
    bra_g::Symbol,
    loss_fn::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int},
)
    ##
    ket_fn = :ket_C_smse
    ket_g = :g_identity
    ##
    dir_model = projectdir("results/HatMatrix/$(name_data)/$(name_model)")
    err_paths = PDEHats.find_files_by_suffix(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv_$(bra_g).jld2",
    )
    ##
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    dir_batch = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    path_err = only(
        filter(
            r -> occursin("seed_$(seed)", r) && occursin(dir_batch, r),
            err_paths,
        ),
    )
    err = load(path_err)["err_eqv"]
    return Float64.(err)
end
##
function get_diffs(
    name_data::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    diff_fn::Symbol=:diff_smse,
)
    ##
    dir_diffs = projectdir("results/Diffs/$(name_data)")
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    name_file_load = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    dir_load = PDEHats.find_files_by_suffix(dir_diffs, name_file_load * ".jld2")
    path_diff = only(
        filter(
            r -> occursin("seed_$(seed)", r) && occursin("$(diff_fn)", r),
            dir_load,
        ),
    )
    diffs = load(path_diff)["diffs"]
    return Float64.(diffs)
end
##
