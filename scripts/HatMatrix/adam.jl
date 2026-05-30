## Saves bra-ket overlap
function save_hat_adam(s::Int)
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
    ##
    bra_fns = [ket_C_smse]
    bra_gs = [
        g_identity,
        get_translations_line()...,
        get_rotations()...,
        get_translations_box()...,
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
                    get_hat_adam(
                        name_model,
                        name_data,
                        seed,
                        epoch,
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
                        get_hat_adam(
                            name_model,
                            name_data,
                            seed,
                            epoch,
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
function get_hat_adam(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    epoch::Int,
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
    dir_results = "results/Adam/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)"
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
    keys_ckpt_to_load = ("chs", "st", "ps", "st_opt", "step")
    path_ckpt = dir_load * "checkpoint_$(epoch).jld2"
    ckpt = PDEHats.load_keys_jld2(path_ckpt, keys_ckpt_to_load)
    chs = ckpt["chs"]
    st = ckpt["st"] |> dev
    ps = ckpt["ps"]
    ps = ComponentArray(ps) |> dev
    st_opt_ckpt = ckpt["st_opt"]
    step_ckpt = ckpt["step"]
    ## Model
    model = PDEHats.get_model(chs, name_model, name_data)
    sm = StatefulLuxLayer{true}(model, ps, st)
    ## TrainState
    opt = AdamW(; eta=5.0f-4, lambda=lambda)
    state_train = Training.TrainState(model, ps, st, opt)
    st_opt_ckpt_state = st_opt_ckpt.state |> dev
    @set! state_train.optimizer_state.state = st_opt_ckpt_state
    @set! state_train.step = step_ckpt
    ## Load Batch
    obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
    (input, target) = obs
    ## Ket
    input_ket_g = ket_g(input)
    target_ket_g = ket_g(target)
    ## Shape
    (Lx, Ly, F, T, B) = size(input_ket_g)
    ## Adam
    backend_ad = AutoZygote()
    loss_fn = PDEHats.loss_mse_scaled
    ket_fn = ket_C_smse
    grads_array = map(Iterators.product(1:T, 1:B)) do (t, b)
        println("Adam: t=$t, b=$b")
        input_t_b =
            reshape(input_ket_g[:, :, :, t, b], (Lx, Ly, F, 1)) |> dev
        target_t_b =
            reshape(target_ket_g[:, :, :, t, b], (Lx, Ly, F, 1)) |> dev
        data = (input_t_b, target_t_b)
        grads, _, _, _ = Lux.Training.compute_gradients(
            backend_ad, loss_fn, data, state_train
        )
        _, grads_adam_lazy = Optimisers.apply!(
            opt,
            deepcopy(state_train.optimizer_state.state),
            deepcopy(ps),
            deepcopy(grads),
        )
        grads_adam = Base.materialize(grads_adam_lazy) |> cpu_device()
        grads_out = deepcopy(collect(grads_adam))
        return grads_out
    end
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
                        J_ket_t1_n1 = grads_array[t1, n1]
                        bra_J_t2_n2 = bra_J_array[t2, n2]
                        return dot(bra_J_t2_n2, J_ket_t1_n1)
                    end
                    dict_result = Dict("bra_J_chi_J_ket" => bra_J_J_ket)
                    return dict_result
                end
        end
    end
    return nothing
end
##
