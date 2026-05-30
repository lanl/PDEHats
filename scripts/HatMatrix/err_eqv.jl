## Saves equivariance errors
function save_err_eqv()
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
    for name_data in names_data
        if name_data == :CE
            epoch = 150
        elseif name_data == :NS
            epoch = 100
        end
        for name_model in names_model
            for seed in seeds
                for idx_NT in idx_NTs
                    get_err_eqv(name_model, name_data, seed, epoch, idx_NT;)
                end
            end
        end
    end
    return nothing
end
##
function get_err_eqv(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    epoch::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    loss_fn=loss_smse,
    ratio_train::AbstractFloat=0.65f0,
    ratio_val::AbstractFloat=0.05f0,
    ket_gs::Vector=[
        g_identity,
        get_translations_line()...,
        get_rotations()...,
        get_translations_box()...,
    ],
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
    dir_model = "results/Eqv/$(name_data)/$(nameof(loss_fn))/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)"
    dir_exp = savename((epoch=epoch, Tmax=T_max); equals="_")
    dir_batch = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    path_save = projectdir(join([dir_model, dir_exp, dir_batch], "/"))
    name_file_save = "err_eqv"
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
    ## Load Batch
    obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
    (input, target) = obs
    # Ket
    err_eqv_array = map(ket_gs) do ket_g
        name_file_save = "err_eqv_$(nameof(ket_g))"
        println("Getting $(name_file_save): $(dir_batch)")
        dict_result, path_dict_result =
            produce_or_load((;), path_save; filename=name_file_save) do _
                input_ket_g = ket_g(input) |> dev
                target_ket_g = ket_g(target) |> dev
                pred_ket_g, _ = model(input_ket_g, ps, st)
                # Errs
                err_eqv =
                    loss_fn(input_ket_g, target_ket_g, pred_ket_g) |>
                    cpu_device()
                dict_result = Dict("err_eqv" => err_eqv)
                return dict_result
            end
        return dict_result["err_eqv"]
    end
    return err_eqv_array
end
##
function loss_smse(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    loss_unscaled = abs2.(target .- pred)
    scale = sqrt.(mean(abs2.(target); dims=(1, 2))) .+ 1.0f-6
    loss = dropdims(
        mean(loss_unscaled ./ scale; dims=(1, 2, 3)); dims=(1, 2, 3)
    )
    return loss
end
function loss_smse(
    target::AbstractArray{Float32,5}, pred::AbstractArray{Float32,5}
)
    loss_unscaled = abs2.(target .- pred)
    scale = sqrt.(mean(abs2.(target); dims=(1, 2))) .+ 1.0f-6
    loss = dropdims(
        mean(loss_unscaled ./ scale; dims=(1, 2, 3)); dims=(1, 2, 3)
    )
    return loss
end
##
