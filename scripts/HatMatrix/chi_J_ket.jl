## Saves "RHS" chi_G_J_ket
function save_chi_g_J_ket()
    ##
    names_data = (:CE, :NS)
    names_model = (:UNet, :ViT)
    ket_fn = ket_C_smse
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    ##
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
            for idx_NT in idx_NTs
                for seed in seeds
                    get_chi_g_J_ket(
                        name_model,
                        name_data,
                        seed,
                        epoch,
                        ket_fn,
                        idx_NT;
                        rtol=rtol,
                    )
                end
            end
        end
    end
    ##
    return nothing
end
##
function get_chi_g_J_ket(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    epoch::Int,
    ket_fn,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    ratio_train::AbstractFloat=0.65f0,
    ratio_val::AbstractFloat=0.05f0,
    ket_g=g_identity,
    lambda::Float32=1.0f-4,
    rtol::Float32=5.0f-2,
    T_max::Int=17,
    itmax::Int=15,
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)

    ## Unpack
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    ## Saving Pathing
    dir_model = "results/HatMatrix/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)"
    dir_exp = savename(
        (epoch=epoch, lambda=PDEHats.float_to_string(lambda), Tmax=T_max);
        equals="_",
    )
    dir_batch = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    dir_save = projectdir(join([dir_model, dir_exp, dir_batch], "/"))
    name_file = "chi_$(nameof(ket_g))_J_$(nameof(ket_fn))"
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
    ## Ket
    input_ket_g = ket_g(input) |> dev
    target_ket_g = ket_g(target) |> dev
    pred_ket_g, _ = model(input_ket_g, ps, st)
    ket = ket_fn(input_ket_g, target_ket_g, pred_ket_g)
    ## Shape
    (Lx, Ly, F, T, B) = size(input_ket_g)
    input_ket_g_r = reshape(input_ket_g, (Lx, Ly, F, T * B))
    ## Construct Matrix-Free Jacobians
    sm = StatefulLuxLayer{true}(model, ps, st)
    sm_input_ket_g_r = Base.Fix1(sm, input_ket_g_r)
    jvp = Jacobian{JVP_FullBatch}(sm_input_ket_g_r, ps, size(input_ket_g))
    vjp = Jacobian{VJP_FullBatch}(sm_input_ket_g_r, ps, size(input_ket_g))
    ## LinearOperator Interface
    n = prod(size(input_ket_g)) * T * B
    m = length(ps) * T * B
    S = ket isa CuArray ? CuVector{Float32} : Vector{Float32}
    jacobian_operator = LinearOperator(
        Float32, m, n, false, false, vjp, jvp, jvp; S=S
    )
    ## Workspace
    workspace = CraigmrWorkspace(m, n, S)
    ## Load/Init Checkpoint
    path_save_ckpt = dir_save * "/checkpoint"
    dict_workspace, path_dict_workspace_ckpt =
        produce_or_load((;), path_save_ckpt; filename=name_file) do _
            # [p * T * B]
            J_ket_initial_r_cpu = get_J_ket(sm, ps, input_ket_g, ket)
            J_ket_initial_r = J_ket_initial_r_cpu |> dev
            # Residual
            norm_residual_initial = norm(J_ket_initial_r_cpu)
            # Solve
            println("Entering Solve")
            craigmr!(
                workspace,
                jacobian_operator,
                J_ket_initial_r;
                λ=sqrt(lambda),
                verbose=1,
                itmax=1,
            )
            # [p * T * B]
            chi_J_ket_r_cpu = Krylov.solution(workspace, 2) |> cpu_device()
            # Ax + (lambda * y) = b
            x_r = Krylov.solution(workspace, 1)
            Jx_r = (jacobian_operator * x_r) |> cpu_device()
            lambda_y_r = lambda .* chi_J_ket_r_cpu
            # Running b
            J_ket_r_cpu = J_ket_initial_r_cpu .- (Jx_r .+ lambda_y_r)
            # Dict
            dict_workspace = Dict(
                "J_ket_r" => J_ket_r_cpu,
                "chi_J_ket_r" => chi_J_ket_r_cpu,
                "norm_residual_initial" => norm_residual_initial,
            )
            #
            return dict_workspace
        end
    ## Krylov
    dir_rtol = savename((rtol=PDEHats.float_to_string(rtol),); equals="_")
    path_save = dir_save * "/" * dir_rtol
    dict_workspace, path_dict_workspace =
        produce_or_load((;), path_save; filename=name_file) do _
            println("Entering Solve")
            is_solved = false
            while ~is_solved
                dict_workspace = update!!(
                    workspace,
                    dict_workspace,
                    jacobian_operator,
                    lambda,
                    rtol,
                    itmax;
                    dev=dev,
                )
                wsave(path_dict_workspace_ckpt, dict_workspace)
                is_solved = Krylov.issolved(workspace)
            end
            ## Update Checkpoint
            tagsave(path_save_ckpt * "/" * name_file * ".jld2", dict_workspace)
            return dict_workspace
        end
    ##
    return dict_workspace
end
##
function update!!(
    workspace::CraigmrWorkspace,
    dict_workspace::Dict,
    jacobian_operator::LinearOperator,
    lambda::Float32,
    rtol::Float32,
    itmax::Int;
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    ## [p * T * B]
    J_ket_r_cpu = dict_workspace["J_ket_r"]
    J_ket_r = J_ket_r_cpu |> dev
    norm_residual_initial = dict_workspace["norm_residual_initial"]
    atol = rtol * norm_residual_initial
    ## Solve
    craigmr!(
        workspace,
        jacobian_operator,
        J_ket_r;
        λ=sqrt(lambda),
        verbose=1,
        atol=atol,
        itmax=itmax,
    )
    ## [p * T * B]
    y_r = Krylov.solution(workspace, 2) |> cpu_device()
    chi_J_ket_r_cpu = dict_workspace["chi_J_ket_r"] .+ y_r
    ## Ax + (lambda * y) = b
    x_r = Krylov.solution(workspace, 1)
    Jx_r = (jacobian_operator * x_r) |> cpu_device()
    lambda_y_r = lambda .* y_r
    ## Running b
    J_ket_r_cpu = J_ket_r_cpu .- (Jx_r .+ lambda_y_r)
    ##
    dict_workspace["J_ket_r"] = J_ket_r_cpu
    dict_workspace["chi_J_ket_r"] = chi_J_ket_r_cpu
    ##
    return dict_workspace
end
##
function get_J_ket(
    sm::StatefulLuxLayer,
    ps::ComponentArray,
    input::AbstractArray{Float32,5},
    ket::AbstractArray{Float32,5},
)
    println("Getting RHS")
    (Lx, Ly, F, T, B) = size(input)
    # Compute b in Ax + (lambda * y) = b
    J_ket_array = map(Iterators.product(1:T, 1:B)) do (t, b)
        ket_t_b = view(ket, :, :, :, (t:t), b)
        input_t_b = view(input, :, :, :, (t:t), b)
        sm_input_t_b = Base.Fix1(sm, input_t_b)
        # [p]
        J_ket_t_b = getdata(
            Lux.vector_jacobian_product(
                sm_input_t_b, AutoZygote(), ps, ket_t_b
            ),
        )
        J_ket_t_b_cpu = J_ket_t_b |> cpu_device()
        return J_ket_t_b_cpu
    end
    # [p, T, B]
    J_ket = stack(J_ket_array)
    # [p * T * B]
    J_ket_r = vec(J_ket)
    return J_ket_r
end
##
