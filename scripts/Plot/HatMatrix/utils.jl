##
function get_hat_normed(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int,
    bra_fn::Symbol,
    bra_g::Symbol,
    c_inds::Vector{<:CartesianIndex};
    normalization::Symbol=:nothing,
)
    ##
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
        (; idx_rp=7, idx_crp=7, idx_rpui=7),
        (; idx_rp=8, idx_crp=8, idx_rpui=8),
        (; idx_rp=9, idx_crp=9, idx_rpui=9),
        (; idx_rp=10, idx_crp=10, idx_rpui=10),
        (; idx_rp=11, idx_crp=11, idx_rpui=11),
        (; idx_rp=12, idx_crp=12, idx_rpui=12),
        (; idx_rp=13, idx_crp=13, idx_rpui=13),
        (; idx_rp=14, idx_crp=14, idx_rpui=14),
        (; idx_rp=15, idx_crp=15, idx_rpui=15),
        (; idx_rp=16, idx_crp=16, idx_rpui=16),
        (; idx_rp=17, idx_crp=17, idx_rpui=17),
        (; idx_rp=18, idx_crp=18, idx_rpui=18),
    )
    ##
    hats_seed_batch = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        hat_normed = get_hat_normed(
            name_model,
            name_data,
            epoch,
            bra_fn,
            bra_g,
            seed,
            idx_NT;
            normalization=normalization,
        )
        hats = map(c -> hat_normed[c], c_inds)
        return hats
    end
    return hats_seed_batch
end
##
function get_hat_normed(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int,
    bra_fn::Symbol,
    bra_g::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    normalization::Symbol=:nothing,
)
    hat = get_hat(name_model, name_data, epoch, bra_fn, bra_g, seed, idx_NT)
    (T, B) = size(hat)[1:2]
    if normalization == :standard
        hat_std = std(hat; dims=(3, 4)) .+ 1.0f-6
        hat_normed = hat ./ hat_std
    elseif normalization == :gibbs
        hat_std = std(hat; dims=(3, 4)) .+ 1.0f-6
        hat_normed = hat ./ hat_std
        hat_normed = softmax(hat_normed; dims=(3, 4))
    elseif normalization == :cosine
        hat_normed =
            map(Iterators.product(1:T, 1:B, 1:T, 1:B)) do (t2, b2, t1, b1)
                g1 = sqrt(hat[t1, b1, t1, b1]) .+ 1.0f-6
                g2 = sqrt(hat[t2, b2, t2, b2]) .+ 1.0f-6
                return hat[t2, b2, t1, b1] / (g1 * g2)
            end
    elseif normalization == :column
        hat_normed =
            map(Iterators.product(1:T, 1:B, 1:T, 1:B)) do (t2, b2, t1, b1)
                g2 = hat[t2, b2, t2, b2] .+ 1.0f-6
                return hat[t2, b2, t1, b1] / g2
            end
    elseif normalization == :row
        hat_normed =
            map(Iterators.product(1:T, 1:B, 1:T, 1:B)) do (t2, b2, t1, b1)
                g1 = hat[t1, b1, t1, b1] .+ 1.0f-6
                return hat[t2, b2, t1, b1] / g1
            end
    elseif normalization == :trace
        hat_r = reshape(hat, (T * B, T * B))
        hat_tr = tr(hat_r) / (T * B)
        hat_normed = hat ./ (hat_tr .+ 1.0f-6)
    elseif normalization == :loss
        # [T, B]
        errs = get_err(name_model, name_data, seed, idx_NT)
        errs_r = reshape(errs, (T, B, 1, 1)) .+ 1.0f-6
        hat_normed = hat ./ errs_r
    else
        hat_normed = hat
    end
    return hat_normed
end
##
function get_hat(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int,
    bra_fn::Symbol,
    bra_g::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
)
    ##
    ket_fn = :loss_mse_scaled
    ket_g = :g_identity
    ##
    dir_results = projectdir("results/bra_chi_ket/$(name_data)/$(name_model)")
    name_chi_ket = "chi_$(ket_g)_$(string(ket_fn))"
    ##
    name_bra = "$(string(bra_fn))_$(string(bra_g))_"
    name_file_save = name_bra * name_chi_ket
    ##
    result_paths = PDEHats.find_files(dir_results, name_file_save, ".jld2")
    filter!(p -> occursin("epoch_$(epoch)_", p), result_paths)
    if (bra_g == :g_identity) && (bra_fn == :loss_mse_scaled)
        filter!(p -> occursin("/epoch/", p), result_paths)
    end
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
    hat = load(path_hat)["bra_chi_ket"]
    return hat
end
##
function get_err(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    ket_g::Symbol=:g_identity,
)
    ##
    loss_fn = :loss_smse
    ##
    err_paths = PDEHats.find_files_by_suffix(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv_$(ket_g).jld2",
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
    return err
end
##
function get_diffs(
    name_data::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    diff_fn::Symbol=:diff_mse,
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
    return diffs
end
##
