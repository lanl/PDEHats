##
function compute_bra_J_g_chi_g_J_ket(
    name_model::Symbol,
    seed::Int,
    N_ckpt::Int,
    ket_fn,
    bra_fns::Vector,
    idx_tuple::NTuple{3,Int};
    ket_g=g_identity,
    bra_gs::Vector=[g_identity],
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
    obs = PDEHats.shift_pair(trajectories)
    # Unpack
    (input, target) = obs
    ## Loading chi_J_ket
    println("Loading chi_J_ket")
    dir_save_results = "results/HatMatrix/$(name_model)/seed_$(seed)/ckpt_$(N_ckpt)/"
    _rtol = string_rtol(rtol)
    dir_save_exp = savename((lambda=lambda, rtol=_rtol, Tmax=T_max); equals="_")
    savename_rhs = "/chi_$(nameof(ket_g))_J_$(nameof(ket_fn))/"
    savename_result = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    path_save = projectdir(
        dir_save_results *
        dir_save_exp *
        savename_rhs *
        savename_result *
        ".jld2",
    )
    chi_g_J_ket = load(path_save)["chi_g_J_ket"] |> dev
    ## Compute/Saving bra_J_g_chi_g_J_ket
    println("Computing Observables")
    for bra_g in bra_gs
        # Bra
        input_bra_g = bra_g(input) |> dev
        target_bra_g = bra_g(target) |> dev
        pred_bra_g, _ = model(input_bra_g, ps, st)
        # Compute J_g_chi_g_J_ket
        J_g_chi_g_J_ket = get_J_chi_J_ket(
            chi_g_J_ket, model, ps, st, input_bra_g
        )
        # Compute/Save bra_J_g_chi_g_J_ket
        for bra_fn in bra_fns
            path_save = projectdir(
                dir_save_results *
                dir_save_exp *
                savename_rhs *
                "$(nameof(bra_fn))_J_$(nameof(bra_g))_chi_$(nameof(ket_g))_J_$(nameof(ket_fn))/" *
                savename_result *
                ".jld2",
            )
            get_bra_J_chi_J_ket(
                input_bra_g,
                target_bra_g,
                pred_bra_g,
                J_g_chi_g_J_ket,
                bra_fn;
                path_save=path_save,
            )
        end
    end
    ##
    return nothing
end
##
function get_J_chi_J_ket(
    chi_J_ket::AbstractArray{Float32,3},
    model::AbstractLuxLayer,
    ps::ComponentArray,
    st::NamedTuple,
    input::AbstractArray{Float32,5};
    dev=gpu_device(),
)
    # Shape
    (Lx, Ly, F, T, B) = size(input)
    input_r = reshape(input, (Lx, Ly, F, T * B))
    # Construct Matrix-Free Jacobians
    sm = StatefulLuxLayer{true}(model, ps, st)
    sm_input_r = Base.Fix1(sm, input_r)
    # [p, T * B]
    chi_J_ket_r = reshape(chi_J_ket, (length(ps), T * B))
    J_chi_J_ket_r_array = map(axes(chi_J_ket_r, ndims(chi_J_ket_r))) do j
        # [p]
        chi_J_ket_r_j = selectdim(chi_J_ket_r, ndims(chi_J_ket_r), j)
        # [Lx, Ly, F, T * B]
        J_chi_J_ket_j_r = getdata(
            Lux.jacobian_vector_product(
                sm_input_r, AutoForwardDiff(), ps, chi_J_ket_r_j
            ),
        )
        # [Lx, Ly, F, T, B]
        J_chi_J_ket_j = reshape(J_chi_J_ket_j_r, (Lx, Ly, F, T, B))
        return J_chi_J_ket_j
    end
    # [Lx, Ly, F, T, B, T * B]
    J_chi_J_ket_r = stack(J_chi_J_ket_r_array)
    # [Lx, Ly, F, T, B, T, B]
    J_chi_J_ket = reshape(J_chi_J_ket_r, (Lx, Ly, F, T, B, T, B))
    return J_chi_J_ket
end
##
function get_bra_J_chi_J_ket(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    J_chi_J_ket::AbstractArray{Float32,7},
    bra_fn;
    path_save::String="dir_save_default/bra_J_chi_J_ket.jld2",
    dev=gpu_device(),
)
    # Compute
    (Lx, Ly, F, T, B, _, _) = size(J_chi_J_ket)
    J_chi_J_ket_r = reshape(J_chi_J_ket, (Lx, Ly, F, T, B, T * B))
    #
    bra_J_chi_J_ket_r_array = map(axes(J_chi_J_ket_r, 6)) do j
        J_chi_J_ket_j = copy(selectdim(J_chi_J_ket_r, 6, j))
        bra_J_chi_J_ket_j = bra_fn(input, target, pred, J_chi_J_ket_j)
        return bra_J_chi_J_ket_j
    end
    # [T, B, T * B]
    bra_J_chi_J_ket_r = stack(bra_J_chi_J_ket_r_array)
    # [T, B, T, B]
    bra_J_chi_J_ket = reshape(bra_J_chi_J_ket_r, (T, B, T, B)) |> cpu_device()
    dict_result = Dict("bra_J_chi_J_ket" => bra_J_chi_J_ket)
    wsave(path_save, dict_result)
    return bra_J_chi_J_ket
end
##
