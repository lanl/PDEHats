## Saves bra-ket overlap
function save_hat_euclidean_corrected(s::Int)
    ##
    names_data = (:CE, :NS)
    names_model = (:UNet, :ViT)
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )[s:s]
    ket_fn = ket_C_smse
    ##
    bra_fns = [ket_C_smse]
    bra_gs = [
        g_identity,
        # get_translations_line()...,
        # get_rotations()...,
        # get_translations_box()...,
    ]
    ##
    for name_data in names_data
        if name_data == :CE
            epoch = 150
        elseif name_data == :NS
            epoch = 100
        end
        for name_model in names_model
            for seed in seeds
                for idx_NT in idx_NTs
                    get_hat_euclidean_corrected(
                        name_model,
                        name_data,
                        seed,
                        epoch,
                        ket_fn,
                        bra_fns,
                        idx_NT;
                        bra_gs=bra_gs,
                    )
                end
            end
        end
        ## Mass/Energy (CE Only)
        if name_data == :CE
            _bra_fns = [ket_C_mass, ket_C_energy]
            _bra_gs = [g_identity]
            for name_model in names_model
                for seed in seeds
                    for idx_NT in idx_NTs
                        get_hat_euclidean_corrected(
                            name_model,
                            name_data,
                            seed,
                            epoch,
                            ket_fn,
                            _bra_fns,
                            idx_NT;
                            bra_gs=_bra_gs,
                        )
                    end
                end
            end
        end
    end
    ##
    return nothing
end
##
function get_hat_euclidean_corrected(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    epoch::Int,
    ket_fn,
    bra_fns::Vector,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    ratio_train::AbstractFloat=0.65f0,
    ratio_val::AbstractFloat=0.05f0,
    ket_g=g_identity,
    bra_gs::Vector=[g_identity],
    lambda::Float32=1.0f-4,
    T_max::Int=17,
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    ## Unpack
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    ## Saving Pathing
    dir_model = "results/EuclideanCorrected/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)"
    dir_exp = savename(
        (epoch=epoch, lambda=PDEHats.float_to_string(lambda), Tmax=T_max);
        equals="_",
    )
    dir_batch = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    dir_save = projectdir(join([dir_model, dir_exp, dir_batch], "/"))
    ## Load Pathing
    dir_load = projectdir(
        "results/Train/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)/",
    )
    ## CKPT
    keys_ckpt_to_load = ("chs", "st", "ps")
    path_ckpt = dir_load * "checkpoint_$(epoch).jld2"
    ckpt = PDEHats.load_keys_jld2(path_ckpt, keys_ckpt_to_load)
    chs = ckpt["chs"]
    st = ckpt["st"] |> dev
    ps = ckpt["ps"]
    ps = ComponentArray(ps) |> dev
    ## Model
    model = PDEHats.get_model(chs, name_model, name_data)
    sm = StatefulLuxLayer{true}(model, ps, st)
    ## Load Batch
    obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
    (input, target) = obs
    ## Ket
    input_ket_g = ket_g(input) |> dev
    target_ket_g = ket_g(target) |> dev
    pred_ket_g = sm(input_ket_g)
    ket = ket_fn(input_ket_g, target_ket_g, pred_ket_g)
    ## Shape
    (Lx, Ly, F, T, B) = size(input_ket_g)
    # [p,  T, B]
    J_ket_array = get_vjp(sm, ps, input_ket_g, ket)
    ## Produce
    for bra_fn in bra_fns
        for bra_g in bra_gs
            name_file = "$(nameof(bra_fn))_J_$(nameof(bra_g))_chi_$(nameof(ket_g))_J_$(nameof(ket_fn))"
            dict_result, path_dict_result =
                produce_or_load((;), dir_save; filename=name_file) do _
                    ## Bra
                    input_bra_g = bra_g(input) |> dev
                    target_bra_g = bra_g(target) |> dev
                    pred_bra_g = sm(input_bra_g)
                    bra = bra_fn(input_bra_g, target_bra_g, pred_bra_g)
                    # [p,  T, B]
                    bra_J_array = get_vjp(sm, ps, input_bra_g, bra)
                    bra_J_J_ket = map(
                        Iterators.product(1:T, 1:B, 1:T, 1:B)
                    ) do (t2, n2, t1, n1)
                        gR = J_ket_array[t1, n1]
                        gL = bra_J_array[t2, n2]
                        euclidean = dot(gL, gR)
                        correction = overlap_correct(
                            name_model, name_data, seed, dir_batch, gL, gR
                        )
                        res = euclidean + correction
                        return res
                    end
                    dict_result = Dict("bra_J_chi_J_ket" => bra_J_J_ket)
                    return dict_result
                end
        end
    end
    return nothing
end
##
function overlap_correct(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    dir_batch::String,
    gL::AbstractVector{Float32},
    gR::AbstractVector{Float32},
)
    dir_load = projectdir("results/EigenNTK/$(name_data)/$(name_model)")
    paths_load = PDEHats.find_files_by_suffix(dir_load, "eigen_max.jld2")
    filter!(p -> occursin(dir_batch, p), paths_load)
    filter!(p -> occursin("seed_$(seed)", p), paths_load)
    path_load = only(paths_load)
    # This scale factor previously omitted for numerical reasons
    LLFTB = 128 * 128 * 4 * 16 * 3
    vals = load(path_load)["vals"]
    vecs = load(path_load)["vecs"]
    corrections = map(vals, vecs) do _val, _vec
        return ((LLFTB / _val) - 1) * dot(gL, _vec) * dot(_vec, gR)
    end
    correction = sum(corrections)
    return correction
end
##
