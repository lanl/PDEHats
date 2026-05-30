##
function get_vjp(
    sm::StatefulLuxLayer,
    ps::ComponentArray,
    input::AbstractArray{Float32,5},
    ket::AbstractArray{Float32,5},
)
    println("Getting JVP")
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
    return J_ket_array
end
##
