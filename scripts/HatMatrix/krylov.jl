##
function get_chi_J_ket(
    model::AbstractLuxLayer,
    ps::ComponentArray,
    st::NamedTuple,
    input::AbstractArray{Float32,5},
    ket::AbstractArray{Float32,5},
    lambda::Float32,
    rtol::Float32;
    krylov_parallel::Symbol=:Full,
)
    if krylov_parallel == :Full
        chi_J_ket = get_chi_J_ket_full(model, ps, st, input, ket, lambda, rtol)
    elseif krylov_parallel == :Partial
        chi_J_ket = get_chi_J_ket_partial(
            model, ps, st, input, ket, lambda, rtol
        )
    elseif krylov_parallel == :Serial
        chi_J_ket = get_chi_J_ket_serial(
            model, ps, st, input, ket, lambda, rtol
        )
    else
        throw(
            "only :Full, :Partial, and :Serial krylov_parallel values are supported",
        )
    end
    return chi_J_ket
end
##
function get_chi_J_ket_full(
    model::AbstractLuxLayer,
    ps::ComponentArray,
    st::NamedTuple,
    input::AbstractArray{Float32,5},
    ket::AbstractArray{Float32,5},
    lambda::Float32,
    rtol::Float32,
)
    # Shape
    (Lx, Ly, F, T, B) = size(input)
    input_r = reshape(input, (Lx, Ly, F, T * B))
    ket_r = reshape(ket, (Lx, Ly, F, T * B))
    # Construct Matrix-Free Jacobians
    sm = StatefulLuxLayer{true}(model, ps, st)
    sm_input_r = Base.Fix1(sm, input_r)
    jvp = Jacobian{JVP_Full}(sm_input_r, ps, size(input))
    vjp = Jacobian{VJP_Full}(sm_input_r, ps, size(input))
    # LinearOperator Interface
    n = prod(size(input)) * T * B
    m = length(ps) * T * B
    S = ket isa CuArray ? CuVector{Float32} : Vector{Float32}
    jacobian_operator = LinearOperator(
        Float32, m, n, false, false, vjp, jvp, jvp; S=S
    )
    # Solve
    workspace_craigmr = CraigmrWorkspace(m, n, S)
    # Compute y (The RHS in Ax = b)
    J_ket_array = map(Iterators.product(1:T, 1:B)) do (t, b)
        ket_t_b = view(ket,:,:,:,t,(b:b))
        input_t_b = view(input,:,:,:,t,(b:b))
        sm_input_t_b = Base.Fix1(sm, input_t_b)
        # [p]
        J_ket_t_b = getdata(
            Lux.vector_jacobian_product(
                sm_input_t_b, AutoZygote(), ps, ket_t_b
            ),
        )
        return J_ket_t_b
    end
    # [p, T, B]
    J_ket = stack(J_ket_array)
    # [p * T * B]
    J_ket_r = vec(J_ket)
    craigmr!(
        workspace_craigmr,
        jacobian_operator,
        J_ket_r;
        λ=lambda,
        verbose=1,
        rtol=rtol,
    )
    # [p * T * B]
    _, chi_J_ket_r, stats = Krylov.results(workspace_craigmr)
    @show stats
    # [p, T, B]
    chi_J_ket = reshape(chi_J_ket_r, (length(ps), T, B)) |> cpu_device()
    return chi_J_ket
end
##
function get_chi_J_ket_partial(
    model::AbstractLuxLayer,
    ps::ComponentArray,
    st::NamedTuple,
    input::AbstractArray{Float32,5},
    ket::AbstractArray{Float32,5},
    lambda::Float32,
    rtol::Float32,
)
    # Shape
    (Lx, Ly, F, T, B) = size(input)
    input_r = reshape(input, (Lx, Ly, F, T * B))
    # Construct Matrix-Free Jacobians
    sm = StatefulLuxLayer{true}(model, ps, st)
    sm_input_r = Base.Fix1(sm, input_r)
    jvp = Jacobian{JVP_Partial}(sm_input_r, ps, size(input))
    vjp = Jacobian{VJP_Partial}(sm_input_r, ps, size(input))
    # LinearOperator Interface
    n = prod(size(input)) * B
    m = length(ps) * B
    S = ket isa CuArray ? CuVector{Float32} : Vector{Float32}
    jacobian_operator = LinearOperator(
        Float32, m, n, false, false, vjp, jvp, jvp; S=S
    )
    # Solve
    workspace_craigmr = CraigmrWorkspace(m, n, S)
    chi_J_ket_array = map(1:T) do t
        # Compute y (The RHS in Ax = b)
        ket_t = selectdim(ket, 4, t)
        input_t = selectdim(input, 4, t)
        J_ket_t_array = map(axes(ket_t, ndims(ket_t))) do j
            # Select example j
            ket_t_j = selectdim(ket_t, ndims(ket_t), j:j)
            input_t_j = selectdim(input_t, ndims(input_t), j:j)
            sm_input_t_j = Base.Fix1(sm, input_t_j)
            # [p]
            J_ket_t_j = getdata(
                Lux.vector_jacobian_product(
                    sm_input_t_j, AutoZygote(), ps, ket_t_j
                ),
            )
            return J_ket_t_j
        end
        # [p, B]
        J_ket_t = stack(J_ket_t_array)
        # [p * B]
        J_ket_t_r = vec(J_ket_t)
        craigmr!(
            workspace_craigmr,
            jacobian_operator,
            J_ket_t_r;
            λ=lambda,
            verbose=1,
            rtol=rtol,
        )
        # [p * B]
        _, chi_J_ket_t_r, stats = Krylov.results(workspace_craigmr)
        @show stats
        # [p,  B]
        chi_J_ket_t =
            reshape(copy(chi_J_ket_t_r), (length(ps), B)) |> cpu_device()
        return chi_J_ket_t
    end
    # [p, B, T]
    chi_J_ket_p = stack(chi_J_ket_array)
    # [p, T, B]
    chi_J_ket = permutedims(chi_J_ket_p, (1, 3, 2))
    return chi_J_ket
end
##
function get_chi_J_ket_serial(
    model::AbstractLuxLayer,
    ps::ComponentArray,
    st::NamedTuple,
    input::AbstractArray{Float32,5},
    ket::AbstractArray{Float32,5},
    lambda::Float32,
    rtol::Float32,
)
    # Shape
    (Lx, Ly, F, T, B) = size(input)
    input_r = reshape(input, (Lx, Ly, F, T * B))
    # Construct Matrix-Free Jacobians
    sm = StatefulLuxLayer{true}(model, ps, st)
    sm_input_r = Base.Fix1(sm, input_r)
    jvp = Jacobian{JVP_Serial}(sm_input_r, ps, size(input))
    vjp = Jacobian{VJP_Serial}(sm_input_r, ps, size(input))
    # LinearOperator Interface
    n = prod(size(input))
    m = length(ps)
    S = ket isa CuArray ? CuVector{Float32} : Vector{Float32}
    jacobian_operator = LinearOperator(
        Float32, m, n, false, false, vjp, jvp, jvp; S=S
    )
    # Solve
    workspace_craigmr = CraigmrWorkspace(m, n, S)
    chi_J_ket_array = map(Iterators.product(1:T, 1:B)) do (t, b)
        # Compute y (The RHS in Ax = y)
        ket_tb = view(ket,:,:,:,t,(b:b))
        input_tb = view(input,:,:,:,t,(b:b))
        sm_input_tb = Base.Fix1(sm, input_tb)
        # [p]
        J_ket_tb = getdata(
            Lux.vector_jacobian_product(sm_input_tb, AutoZygote(), ps, ket_tb),
        )
        craigmr!(
            workspace_craigmr,
            jacobian_operator,
            J_ket_tb;
            λ=lambda,
            verbose=1,
            rtol=rtol,
        )
        # [p]
        _, chi_J_ket_tb, stats = Krylov.results(workspace_craigmr)
        @show stats
        ##
        chi_J_ket_tb = copy(chi_J_ket_tb) |> cpu_device()
        ##
        return chi_J_ket_tb
    end
    # [p, T, B]
    chi_J_ket = stack(chi_J_ket_array)
    return chi_J_ket
end
##
