##
function save_diffs(
    name_model::Symbol,
    seed::Int,
    N_ckpt::Int,
    ket_fn,
    idx_tuple::NTuple{3,Int};
    ket_g=g_identity,
    lambda::Float32=1.0f-6,
    rtol::Float32=1.0f-2,
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
    @unpack ratio_train,
    ratio_val, seed, chs, name_model, size_batch, p,
    use_parallel_loading = cfg
    T = T_max - 1
    ## CKPT
    path_ckpt = PDEHats.find_files_by_suffix(dir_load, "checkpoint.jld2")
    ckpt = load(only(path_ckpt))
    chs = ckpt["chs"]
    st = ckpt["st"] |> dev
    ps = ckpt["ps"]
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
    obs = PDEHats.shift_pair(trajectories) |> dev
    # Unpack
    (input, target) = obs
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
        "/diffs/L2/" *
        savename_result *
        ".jld2",
    )
    ## Input-Target Diffs
    idx_tb = axes(input)[[4, 5]]
    diffs = map(Iterators.product(idx_tb..., idx_tb...)) do (T, B, t, b)
        input_tb = view(input,:,:,:,t,b)
        target_tb = view(input,:,:,:,T,B)
        return mean(abs2.(input_tb .- target_tb))
    end
    dict_diffs = Dict("diffs" => diffs)
    safesave(path_save, dict_diffs)
    ##
    return nothing
end
##
function save_diffs()
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(2, 2, 2), (3, 3, 3), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(2, 2, 2), (3, 3, 3), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 35
    rtol = 5.0f-2
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 35
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 42
    rtol = 5.0f-2
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (2, 2, 2), (4, 4, 4)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 42
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (4, 4, 4), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(2, 2, 2), (5, 5, 5), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(2, 2, 2), (5, 5, 5), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 35
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 35
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 42
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (5, 5, 5), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 42
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (5, 5, 5), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        save_diffs(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ##
    return nothing
end
##
