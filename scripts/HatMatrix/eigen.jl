## Seeks eigenvalue of eta
function save_chi_qr_eigen(I::Vector{Int})
    ##
    names_data = (:CE, :NS)
    names_model = (:ViT, :UNet)
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
    )[I]
    ##
    for name_data in names_data
        for name_model in names_model
            if name_data == :CE
                if name_model == :UNet
                    epochs = [1, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
                elseif name_model == :ViT
                    epochs = [
                        1, 25, 50, 75, 100, 101, 105, 120, 130, 131, 135, 150
                    ]
                end
            elseif name_data == :NS
                if name_model == :UNet
                    epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                elseif name_model == :ViT
                    epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                end
            end
            for idx_NT in idx_NTs
                for seed in seeds
                    for epoch in epochs
                        save_chi_qr_eigen(
                            name_model, name_data, seed, epoch, idx_NT
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
function save_chi_qr_eigen(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    epoch::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    ratio_train::AbstractFloat=0.65f0,
    ratio_val::AbstractFloat=0.05f0,
    T_max::Int=17,
    lambda::Float32=1.0f-4,
    rtol::Float32=1.0f-6,
    itmax::Int=0,
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    ## Unpack
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    ## Saving Pathing
    dir_results = joinpath(
        "results",
        "Eigen",
        "$(name_data)",
        "$(name_model)",
        "epoch",
        "seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)",
    )
    dir_exp = savename(
        (epoch=epoch, lambda=PDEHats.float_to_string(lambda), Tmax=T_max);
        equals="_",
    )
    dir_batch = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    dir_save = projectdir(join([dir_results, dir_exp, dir_batch], "/"))
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
    opt = Descent(; eta=1)
    state_train = Training.TrainState(model, ps, st, opt)
    ## Load Batch
    obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
    ##
    save_chi_qr_eigen(
        name_data,
        obs,
        state_train;
        lambda=lambda,
        rtol=rtol,
        itmax=itmax,
        dir_save=dir_save,
        dev=dev,
    )
    ##
    return nothing
end
##
function save_chi_qr_eigen(
    name_data::Symbol,
    obs::Tuple,
    state_train::Training.TrainState;
    lambda::Float32=1.0f-4,
    rtol::Float32=1.0f-6,
    itmax::Int=0,
    dir_save::String=projectdir("dir_save_default"),
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    ## Cost Functions
    ket_fn = PDEHats.loss_mse_scaled
    bra_fn = PDEHats.loss_mse_scaled
    ket_g = g_identity
    bra_g = g_identity
    ## Saving
    dir_rtol = savename((rtol=PDEHats.float_to_string(rtol),); equals="_")
    path_save = joinpath(dir_save, dir_rtol)
    name_chi_ket = "chi_$(nameof(ket_g))_$(string(ket_fn))"
    name_bra = "$(nameof(bra_fn))_$(nameof(bra_g))_"
    name_file_save = name_bra * name_chi_ket
    ##
    dict_chi, path_dict_chi =
        produce_or_load((;), path_save; filename=name_file_save) do _
            ## Data
            input = first(obs)
            (Lx, Ly, F, T, B) = size(input)
            input_r = reshape(input, (Lx, Ly, F, T * B))
            ## Unpack
            model = state_train.model
            ps = state_train.parameters
            st = Lux.testmode(state_train.states)
            ## Gradients
            grads_array = get_grads_array(
                ket_fn, ket_g, obs, state_train; dev=dev
            )
            ## QR
            Q_cpu = get_Q_grads(grads_array)
            ## Project
            grads_array_projected = get_grads_array_projected(
                grads_array, Q_cpu
            )
            ##  Model
            sm = StatefulLuxLayer{true}(model, ps, st)
            sm_input_r = Base.Fix1(sm, input_r |> dev)
            ## Compute
            JQ_array = get_JQ_array(Q_cpu, sm_input_r, ps; dev=dev)
            JQ = stack(vec(JQ_array))
            ##
            (FN, M) = size(JQ)
            QJJQ = transpose(JQ) * JQ ./ FN
            QJJQ_cpu = QJJQ |> cpu_device()
            QJJQ_eigen = eigen(QJJQ_cpu)
            vals = QJJQ_eigen.values
            vecs = QJJQ_eigen.vectors
            dict_eigen = Dict("vals" => vals, "vecs" => vecs)
            ##
            return dict_eigen
        end
    ##
    return nothing
end
