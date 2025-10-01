##
function compute_eqv_fwd(
    name_model::Symbol,
    seed::Int,
    N_ckpt::Int,
    ket_fn,
    idx_tuple::NTuple{3,Int};
    ket_gs=[
        g_identity,
        get_translations_line()...,
        get_rotations()...,
        get_translations(; L=16)...,
    ],
    lambda::Float32=1.0f-6,
    rtol::Float32=5.0f-3,
    T_max::Int=17,
    dev=gpu_device(),
)
    ## Load Dir
    dir_load = projectdir(
        "results/Train/$(name_model)/seed_$(seed)/ckpt_$(N_ckpt)/"
    )
    ## CFG
    dir_load_cfg = join(split(dir_load, "/")[1:(end - 2)], "/") * "/ckpt_0/"
    cfg_path = only(PDEHats.find_files_by_suffix(dir_load_cfg, "cfg.jld2"))
    cfg = load(cfg_path)
    @unpack ratio_train, ratio_val, seed, chs, name_model, size_batch, p = cfg
    ## CKPT
    path_ckpt = PDEHats.find_files_by_suffix(dir_load, "checkpoint.jld2")
    ckpt = load(only(path_ckpt))
    chs = ckpt["chs"]
    st = ckpt["st"] |> dev
    ps = ckpt["ps"];
    ps = ComponentArray(ps) |> dev
    ## Seeding
    rng = Xoshiro(seed)
    ## Model
    model = PDEHats.get_model(chs, name_model)
    ## Dataset
    println("Loading Data")
    _, _, dataset_test = PDEHats.get_datasets(
        Lux.replicate(rng), ratio_train, ratio_val; T_max=T_max, p=p
    )
    ## Trajectories
    trajectories_rp = reshape(
        dataset_test.trajectories_rp_r, dataset_test.size_rp
    )
    trajectories_crp = reshape(
        dataset_test.trajectories_crp_r, dataset_test.size_crp
    )
    trajectories_rpui = reshape(
        dataset_test.trajectories_rpui_r, dataset_test.size_rpui
    )
    ## Batch
    idx_rp = idx_tuple[1]
    idx_crp = idx_tuple[2]
    idx_rpui = idx_tuple[3]
    trajectories = [
        copy(selectdim(trajectories_rp, 5, idx_rp:idx_rp)),
        copy(selectdim(trajectories_crp, 5, idx_crp:idx_crp)),
        copy(selectdim(trajectories_rpui, 5, idx_rpui:idx_rpui)),
    ]
    obs = PDEHats.shift_pair(trajectories)
    # Unpack
    (input, target) = obs
    # Ket
    for ket_g in ket_gs
        input_ket_g = ket_g(input) |> dev
        target_ket_g = ket_g(target) |> dev
        pred_ket_g, _ = model(input_ket_g, ps, st)
        # Errs
        err_eqv = _loss_mse_scaled(target_ket_g, pred_ket_g) |> cpu_device()
        dict_result = Dict(string(nameof(ket_g)) => err_eqv)
        ## Paths
        dir_save_model = "results/HatMatrix/$(name_model)/seed_$(seed)/ckpt_$(N_ckpt)/"
        dir_save_exp = savename(
            (lambda=lambda, rtol=string_rtol(rtol), Tmax=T_max); equals="_"
        )
        savename_rhs = "/chi_$(nameof(ket_g))_J_$(nameof(ket_fn))/"
        savename_result = savename(
            (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
        )
        path_save = projectdir(
            dir_save_model *
            dir_save_exp *
            savename_rhs *
            "/errs/mse_scaled/" *
            savename_result *
            ".jld2",
        )
        wsave(path_save, dict_result)
    end
    return nothing
end
##
function _loss_mse_scaled(
    target::AbstractArray{Float32,5}, pred::AbstractArray{Float32,5}
)
    loss_unscaled = abs2.(target .- pred)
    scale = sqrt.(mean(abs2.(target); dims=(1, 2))) .+ 1.0f-6
    loss = dropdims(
        mean(loss_unscaled ./ scale; dims=(1, 2, 5)); dims=(1, 2, 5)
    )
    return loss
end
