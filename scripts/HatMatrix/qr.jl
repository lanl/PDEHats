##
function save_chi_qr()
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
    )
    reverse_epochs = false
    ## F/R
    seeds = reverse(seeds)
    names_model = reverse(names_model)
    names_data = reverse(names_data)
    idx_NTs = reverse(idx_NTs)
    reverse_epochs = true
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
                    if reverse_epochs
                        epochs = reverse(epochs)
                    end
                    for epoch in epochs
                        save_chi_qr(name_model, name_data, seed, epoch, idx_NT)
                    end
                end
            end
        end
    end
    ##
    return nothing
end
## Saves (T, B) Influence Pi
function save_chi_qr(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    epoch::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    ratio_train::AbstractFloat=0.65f0,
    ratio_val::AbstractFloat=0.05f0,
    T_max::Int=17,
    lambda::Float32=1.0f-4,
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    println("$(name_model), $(name_data), $(seed), $(epoch)")
    ## Unpack
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    ## Saving Pathing
    dir_results = joinpath(
        "results",
        "bra_chi_ket",
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
    save_chi_qr(name_data, obs, state_train; dir_save=dir_save, dev=dev)
    ##
    return nothing
end
##
function save_chi_qr(
    name_data::Symbol,
    obs::Tuple,
    state_train::Training.TrainState;
    dir_save::String=projectdir("dir_save_default"),
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    ##
    bra_fn = PDEHats.loss_mse_scaled
    ket_fn = PDEHats.loss_mse_scaled
    ket_g = g_identity
    bra_g = g_identity
    ## Saving
    name_chi_ket = "chi_$(nameof(ket_g))_$(nameof(ket_fn))"
    name_bra = "$(nameof(bra_fn))_$(nameof(bra_g))_"
    name_file_save = name_bra * name_chi_ket
    ## Get Pi
    dict_chi, path_dict_chi =
        produce_or_load((;), dir_save; filename=name_file_save) do _
            ## Data
            input = first(obs)
            (Lx, Ly, F, T, B) = size(input)
            input_r = reshape(input, (Lx, Ly, F, T * B))
            FN = Float32(prod(size(input)))
            ## Unpack
            model = state_train.model
            ps = state_train.parameters
            st = Lux.testmode(state_train.states)
            ##  Model
            sm = StatefulLuxLayer{true}(model, ps, st)
            sm_input_r = Base.Fix1(sm, input_r |> dev)
            ## Gradients
            grads_stack = get_grads_stack(
                ket_fn, ket_g, obs, state_train; dev=dev
            )
            ## SVD
            Q, grads_projected = get_Q_grads_projected(grads_stack)
            ## Jacobian
            JQ = get_JQ(Q, sm_input_r, ps; dev=dev)
            ## Metric
            epsilon = 1.0f-6
            F = svd(JQ; alg=LinearAlgebra.QRIteration())
            S2_eps = ((F.S ./ sqrt(FN)) .^ 2) .+ epsilon
            chi = (F.V) * Diagonal(inv.(S2_eps)) * (F.Vt)
            ## Get Pi
            Q_chi_ket = chi * grads_projected
            bra_chi_ket_r = transpose(grads_projected) * Q_chi_ket
            bra_chi_ket = reshape(bra_chi_ket_r, (T, B, T, B))
            ##
            dict_chi = Dict("bra_chi_ket" => bra_chi_ket)
            return dict_chi
        end
    ##
    return nothing
end
##
function get_JQ(
    Q, sm_input_r, ps; dev::MLDataDevices.AbstractDevice=gpu_device()
)
    M = size(Q, 2)
    JQ_array = map(1:M) do m
        v = Q[:, m] |> dev
        # [Lx, Ly, F, T * B]
        J_v = getdata(
            Lux.jacobian_vector_product(sm_input_r, AutoForwardDiff(), ps, v),
        )
        J_v_r = vec(J_v)
        return J_v_r |> cpu_device()
    end
    JQ = stack(vec(JQ_array))
    return JQ
end
##
function get_grads_stack(
    loss_fn::G,
    g_fn::E,
    obs::Tuple,
    state_train::Training.TrainState;
    dev::MLDataDevices.AbstractDevice=gpu_device(),
) where {G,E}
    (input, target) = obs
    (Lx, Ly, F, T, B) = size(input)
    grads_array = map(Iterators.product(1:T, 1:B)) do (t, b)
        input_tb = g_fn(input[:, :, :, t:t, b:b]) |> dev
        target_tb = g_fn(target[:, :, :, t:t, b:b]) |> dev
        obs_tb = (input_tb, target_tb)
        grads, loss, _, _ = Training.compute_gradients(
            AutoZygote(), loss_fn, obs_tb, state_train
        )
        grads_out = getdata(deepcopy(grads))
        return grads_out |> cpu_device()
    end
    grads_stack = stack(grads_array)
    grads_stack_r = reshape(grads_stack, (:, T * B))
    return grads_stack_r
end
##
function get_Q_grads_projected(grads_stack::Matrix{Float32})
    F = svd(grads_stack; alg=LinearAlgebra.QRIteration())

    s_max = get_s_max(F.S)

    Q = F.U[:, 1:s_max]
    grads_projected = (Diagonal(F.S) * F.Vt)[1:s_max, :]

    return Q, grads_projected
end
##
function get_s_max(S::Vector{Float32}; epsilon::Float32=1.0f-4)
    s2 = S .^ 2
    total = sum(s2)

    if iszero(total)
        return length(S)
    end

    p = cumsum(s2) ./ total

    s_max = findfirst(x -> 1 - x <= epsilon, p)

    if isnothing(s_max)
        s_max = length(S)
    end

    return s_max
end
