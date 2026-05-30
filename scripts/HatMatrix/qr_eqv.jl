## Saves (T, B) Influence Pi
function save_chi_qr_eqv(
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
    reverse_bra_gs::Bool=false,
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
        "bra_chi_ket",
        "$(name_data)",
        "$(name_model)",
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
    opt = Descent(; eta=1.0f0)
    state_train = Training.TrainState(model, ps, st, opt)
    ## Load Batch
    obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
    ##
    ket_fn = PDEHats.loss_mse_scaled
    ket_g = g_identity
    if name_data == :CE
        bra_fns = [
            PDEHats.loss_mse_scaled, PDEHats.loss_mass, PDEHats.loss_energy
        ]
    elseif name_data == :NS
        bra_fns = [PDEHats.loss_mse_scaled]
    end
    ##
    for bra_fn in bra_fns
        if bra_fn == PDEHats.loss_mse_scaled
            bra_gs = [
                g_identity,
                get_rotations()...,
                get_translations_line()...,
                get_translations_box()...,
            ]
        else
            bra_gs = [g_identity]
        end
        if reverse_bra_gs
            bra_gs = reverse(bra_gs)
        end
        for bra_g in bra_gs
            println("$(name_model), $(name_data), $(seed), $(bra_fn), $(bra_g)")
            save_chi_qr_eqv(
                name_data,
                ket_fn,
                bra_fn,
                ket_g,
                bra_g,
                obs,
                state_train;
                dir_save=dir_save,
                dev=dev,
            )
        end
    end
    ##
    return nothing
end
##
function save_chi_qr_eqv(
    name_data::Symbol,
    ket_fn::G1,
    bra_fn::G2,
    ket_g::G3,
    bra_g::G4,
    obs::Tuple,
    state_train::Training.TrainState;
    dir_save::String=projectdir("dir_save_default"),
    dev::MLDataDevices.AbstractDevice=gpu_device(),
) where {G1,G2,G3,G4}
    ## Saving
    name_chi_ket = "chi_$(nameof(ket_g))_$(nameof(ket_fn))"
    name_bra = "$(nameof(bra_fn))_$(nameof(bra_g))_"
    name_file_save = name_bra * name_chi_ket
    ## Compute
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
            ket_grads_stack = get_grads_stack(
                ket_fn, ket_g, obs, state_train; dev=dev
            )
            bra_grads_stack = get_grads_stack(
                bra_fn, bra_g, obs, state_train; dev=dev
            )
            ## QR
            Q, ket_grads_projected, bra_grads_projected = get_Q_grads_projected(
                ket_grads_stack, bra_grads_stack
            )
            ## Compute
            JQ = get_JQ(Q, sm_input_r, ps; dev=dev)
            ## Metric
            epsilon = 1.0f-6
            F = svd(JQ; alg=LinearAlgebra.QRIteration())
            S2_eps = ((F.S ./ sqrt(FN)) .^ 2) .+ epsilon
            chi = (F.V) * Diagonal(inv.(S2_eps)) * (F.Vt)
            ## Get Pi
            Q_chi_ket = chi * ket_grads_projected
            bra_chi_ket_r = transpose(bra_grads_projected) * Q_chi_ket
            bra_chi_ket = reshape(bra_chi_ket_r, (T, B, T, B))
            ##
            dict_chi = Dict("bra_chi_ket" => bra_chi_ket)
            return dict_chi
        end
    ##
    return nothing
end
##
function get_Q_grads_projected(
    ket_grads_stack::Matrix{Float32}, bra_grads_stack::Matrix{Float32}
)
    N_ket = size(ket_grads_stack, 2)
    N_bra = size(bra_grads_stack, 2)
    grads_stack = cat(ket_grads_stack, bra_grads_stack; dims=2)
    F = svd(grads_stack; alg=LinearAlgebra.QRIteration())

    s_max = get_s_max(F.S)

    Q = F.U[:, 1:s_max]
    grads_projected = (Diagonal(F.S) * F.Vt)[1:s_max, :]
    ket_grads_projected = grads_projected[:, 1:N_ket]
    bra_grads_projected = grads_projected[:, (N_ket + 1):(2 * N_bra)]

    return Q, ket_grads_projected, bra_grads_projected
end
