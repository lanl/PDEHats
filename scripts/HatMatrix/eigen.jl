## Seeks eigenvalue of eta
function save_eigen()
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
    for name_data in names_data
        if name_data == :CE
            epoch = 150
        elseif name_data == :NS
            epoch = 100
        end
        for name_model in names_model
            for seed in seeds
                for idx_NT in idx_NTs
                    get_eigen(name_model, name_data, seed, epoch, idx_NT)
                end
            end
        end
    end
    return nothing
end
##
function get_eigen(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    epoch::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    ratio_train::AbstractFloat=0.65f0,
    ratio_val::AbstractFloat=0.05f0,
    howmany::Int=1,
    ket_g=g_identity,
    tol::Float32=1.0f-2,
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
    dir_model = "results/Eigen/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)"
    dir_exp = savename((epoch=epoch, Tmax=T_max); equals="_")
    dir_batch = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    dir_tol = savename(
        (tol=PDEHats.float_to_string(tol), howmany=howmany); equals="_"
    )
    dir_save = projectdir(join([dir_model, dir_exp, dir_batch], "/"))
    path_save = join([dir_save, string(nameof(ket_g)), dir_tol], "/")
    name_file = "eigen"
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
    (input, _) = obs
    input_ket_g = ket_g(input) |> dev
    ## Ket
    input_ket_g = ket_g(input) |> dev
    ## Shape
    (Lx, Ly, F, T, B) = size(input_ket_g)
    input_ket_g_r = reshape(input_ket_g, (Lx, Ly, F, T * B))
    ## Construct Matrix-Free Jacobians
    sm = StatefulLuxLayer{true}(model, ps, st)
    sm_input_ket_g_r = Base.Fix1(sm, input_ket_g_r)
    jvp = Jacobian{JVP}(sm_input_ket_g_r, ps, size(input_ket_g))
    vjp = Jacobian{VJP}(sm_input_ket_g_r, ps, size(input_ket_g))
    ## LinearOperator Interface
    n = prod(size(input_ket_g))
    m = length(ps)
    S = input_ket_g isa CuArray ? CuVector{Float32} : Vector{Float32}
    jacobian_operator = LinearOperator(
        Float32, m, n, false, false, vjp, jvp, jvp; S=S
    )
    eta = transpose(jacobian_operator) * jacobian_operator
    x_0 = randn(Float32, n) |> dev
    ## Solve
    dict_eigen, path_dict_eigen =
        produce_or_load((;), path_save; filename=name_file) do _
            #
            vals, vecs, info = eigsolve(
                x -> eta * x,
                x_0,
                howmany;
                tol=tol,
                verbosity=4,
                issymmetric=true,
            )
            # Results
            vals_cpu = vals |> cpu_device()
            vecs_cpu = vecs |> cpu_device()
            dict_eigen = Dict("vals" => vals_cpu, "vecs" => vecs_cpu)
            #
            return dict_eigen
        end
    ##
    return dict_eigen
end
##
