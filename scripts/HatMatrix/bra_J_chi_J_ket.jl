## Saves bra-ket overlap
function save_bra_J_g_chi_g_J_ket()
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
    )
    ket_fn = ket_C_smse
    ##
    bra_fns = [bra_C_smse]
    bra_gs = [
        g_identity,
        get_translations_line()...,
        get_rotations()...,
        get_translations_box()...,
    ]
    for name_data in names_data
        if name_data == :CE
            epoch = 150
        elseif name_data == :NS
            epoch = 100
        end
        for name_model in names_model
            if name_model == :ViT
                rtol = 5.0f-2
            elseif name_model == :UNet
                rtol = 1.5f-2
            end
            for seed in seeds
                for idx_NT in idx_NTs
                    get_bra_J_g_chi_g_J_ket(
                        name_model,
                        name_data,
                        seed,
                        epoch,
                        ket_fn,
                        bra_fns,
                        idx_NT;
                        bra_gs=bra_gs,
                        rtol=rtol,
                    )
                end
            end
        end
        ## Mass/Energy (CE Only)
        if name_data == :CE
            bra_fns = [bra_C_mass, bra_C_energy]
            bra_gs = [g_identity]
            for name_model in names_model
                if name_model == :ViT
                    rtol = 5.0f-2
                elseif name_model == :UNet
                    rtol = 1.5f-2
                end
                for seed in seeds
                    for idx_NT in idx_NTs
                        get_bra_J_g_chi_g_J_ket(
                            name_model,
                            name_data,
                            seed,
                            epoch,
                            ket_fn,
                            bra_fns,
                            idx_NT;
                            bra_gs=bra_gs,
                            rtol=rtol,
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
function get_bra_J_g_chi_g_J_ket(
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
    rtol::Float32=5.0f-2,
    T_max::Int=17,
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    ## Unpack
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    ## Loading Model
    dir_model = "results/HatMatrix/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)"
    dir_exp = savename(
        (epoch=epoch, lambda=PDEHats.float_to_string(lambda), Tmax=T_max);
        equals="_",
    )
    dir_batch = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    dir_load_model = projectdir(
        "results/Train/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)/",
    )
    ## Loading Chi
    dir_rtol = savename((rtol=PDEHats.float_to_string(rtol),); equals="_")
    dir_load_chi_J_ket = projectdir(
        join([dir_model, dir_exp, dir_batch, dir_rtol], "/")
    )
    name_file_load = "chi_$(nameof(ket_g))_J_$(nameof(ket_fn))"
    path_file_load = dir_load_chi_J_ket * "/" * name_file_load * ".jld2"
    ## CKPT
    keys_ckpt_to_load = ("chs", "st", "ps")
    path_ckpt = dir_load_model * "checkpoint_$(epoch).jld2"
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
    ## Loading chi_J_ket
    println("Loading chi_J_ket")
    dict_workspace = load(projectdir(path_file_load))
    chi_J_ket_r_cpu = dict_workspace["chi_J_ket_r"]
    chi_J_ket_r = chi_J_ket_r_cpu |> dev
    ## Compute/Saving bra_J_g_chi_g_J_ket
    println("Getting Observables")
    bra_fns_gs_J_chi_J_ket = map(bra_gs) do bra_g
        println("Getting bra_g: $(nameof(bra_g))")
        # Compute/Save bra_J_g_chi_g_J_ket
        bra_fns_J_chi_J_ket = map(bra_fns) do bra_fn
            println("Getting bra_fn: $(nameof(bra_fn))")
            name_file_save =
                "$(nameof(bra_fn))_J_$(nameof(bra_g))_" * name_file_load
            path_save = projectdir(dir_load_chi_J_ket * "/observables/")
            ##
            dict_result, path_dict_result = produce_or_load(
                (;), path_save; filename=name_file_save
            ) do _
                # Bra
                input_bra_g = bra_g(input) |> dev
                target_bra_g = bra_g(target) |> dev
                pred_bra_g, _ = model(input_bra_g, ps, st)
                # Compute J_g_chi_g_J_ket
                J_g_chi_J_ket = get_J_chi_J_ket(
                    chi_J_ket_r, model, ps, st, input_bra_g
                )
                bra_J_g_chi_J_ket = get_bra_J_chi_J_ket(
                    input_bra_g,
                    target_bra_g,
                    pred_bra_g,
                    J_g_chi_J_ket,
                    bra_fn,
                )
                dict_result = Dict("bra_J_chi_J_ket" => bra_J_g_chi_J_ket)
                return dict_result
            end
            bra_J_chi_J_ket = dict_result["bra_J_chi_J_ket"]
            return bra_J_chi_J_ket
        end
        return bra_fns_J_chi_J_ket
    end
    ##
    return bra_fns_gs_J_chi_J_ket
end
##
function get_J_chi_J_ket(
    chi_J_ket_r::AbstractArray{Float32,1},
    model::AbstractLuxLayer,
    ps::ComponentArray,
    st::NamedTuple,
    input::AbstractArray{Float32,5};
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    # Shape
    (Lx, Ly, F, T, B) = size(input)
    input_r = reshape(input, (Lx, Ly, F, T * B))
    # Construct Matrix-Free Jacobians
    sm = StatefulLuxLayer{true}(model, ps, st)
    sm_input_r = Base.Fix1(sm, input_r)
    # [p, T, B]
    chi_J_ket = reshape(chi_J_ket_r, (length(ps), T, B))
    #
    J_chi_J_ket_array = map(Iterators.product(1:T, 1:B)) do (t, b)
        # [p]
        chi_J_ket_t_b = view(chi_J_ket, :, t, b)
        # [Lx, Ly, F, T * B]
        J_chi_J_ket_t_b_r = getdata(
            Lux.jacobian_vector_product(
                sm_input_r, AutoForwardDiff(), ps, chi_J_ket_t_b
            ),
        )
        # [Lx, Ly, F, T1, B1]
        J_chi_J_ket_t_b = reshape(J_chi_J_ket_t_b_r, (Lx, Ly, F, T, B))
        return J_chi_J_ket_t_b
    end
    # [Lx, Ly, F, T1, B1, T2, B2]
    J_chi_J_ket = stack(J_chi_J_ket_array)
    return J_chi_J_ket
end
##
function get_bra_J_chi_J_ket(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    J_chi_J_ket::AbstractArray{Float32,7},
    bra_fn;
)
    # Compute
    (Lx, Ly, F, T, B, _, _) = size(J_chi_J_ket)
    #
    bra_J_chi_J_ket_r_array = map(Iterators.product(1:T, 1:B)) do (t, b)
        J_chi_J_ket_t_b = view(J_chi_J_ket, :, :, :, :, :, t, b)
        bra_J_chi_J_ket_t_b = bra_fn(input, target, pred, J_chi_J_ket_t_b)
        bra_J_chi_J_ket_t_b_cpu = bra_J_chi_J_ket_t_b |> cpu_device()
        return bra_J_chi_J_ket_t_b_cpu
    end
    # [T, B, T, B]
    bra_J_chi_J_ket = stack(bra_J_chi_J_ket_r_array)
    return bra_J_chi_J_ket
end
##
