##
abstract type AbstractJacobian end
struct JVP_Full <: AbstractJacobian end
struct VJP_Full <: AbstractJacobian end
struct JVP_Partial <: AbstractJacobian end
struct VJP_Partial <: AbstractJacobian end
struct JVP_Serial <: AbstractJacobian end
struct VJP_Serial <: AbstractJacobian end
##
@concrete struct Jacobian{J<:AbstractJacobian}
    sm_x_r
    ps
    size_x
end
##
function (J::Jacobian{VJP_Full})(
    res::AbstractVector{R}, v_r::AbstractVector{R}, α::R, β::R
) where {R}
    (Lx, Ly, F, T, B) = J.size_x
    # [Lx, Ly, F, T * B, T * B]
    v = reshape(v_r, (Lx, Ly, F, T * B, T * B))
    J_v_array = map(axes(v, ndims(v))) do b
        # [Lx, Ly, F, T * B]
        v_b = selectdim(v, ndims(v), b)
        # [p]
        J_v_b = getdata(
            Lux.vector_jacobian_product(J.sm_x_r, AutoZygote(), J.ps, v_b)
        )
        return J_v_b
    end
    # [p, T * B]
    J_v = stack(J_v_array)
    # [p * T * B]
    J_v_r = vec(J_v)

    if β === zero(β)
        @inbounds res .= α .* J_v_r
    else
        @inbounds res .= α .* J_v_r .+ β .* res
    end

    return res
end
function (J::Jacobian{JVP_Full})(
    res::AbstractVector{R}, v_r::AbstractVector{R}, α::R, β::R
) where {R}
    (Lx, Ly, F, T, B) = J.size_x
    # [p, T * B]
    v = reshape(v_r, (length(J.ps), T * B))
    J_v_array = map(axes(v, ndims(v))) do b
        # [p]
        v_b = selectdim(v, ndims(v), b)
        # [Lx, Ly, F, T * B]
        J_v_b = getdata(
            Lux.jacobian_vector_product(J.sm_x_r, AutoForwardDiff(), J.ps, v_b),
        )
        # [Lx * Ly * F * T * B]
        J_v_b_r = vec(J_v_b)
        return J_v_b_r
    end
    # [Lx * Ly * F * T * B, T * B]
    J_v = stack(J_v_array)
    # [Lx * Ly * F * T * B * T * B]
    J_v_r = vec(J_v)

    if β === zero(β)
        @inbounds res .= α .* J_v_r
    else
        @inbounds res .= α .* J_v_r .+ β .* res
    end

    return res
end
##
function (J::Jacobian{VJP_Partial})(
    res::AbstractVector{R}, v_r::AbstractVector{R}, α::R, β::R
) where {R}
    (Lx, Ly, F, T, B) = J.size_x
    # [Lx, Ly, F, T * B, B]
    v = reshape(v_r, (Lx, Ly, F, T * B, B))
    J_v_array = map(axes(v, ndims(v))) do b
        # [Lx, Ly, F, T * B]
        v_b = selectdim(v, ndims(v), b)
        # [p]
        J_v_b = getdata(
            Lux.vector_jacobian_product(J.sm_x_r, AutoZygote(), J.ps, v_b)
        )
        return J_v_b
    end
    # [p, B]
    J_v = stack(J_v_array)
    # [p * B]
    J_v_r = vec(J_v)

    if β === zero(β)
        @inbounds res .= α .* J_v_r
    else
        @inbounds res .= α .* J_v_r .+ β .* res
    end

    return res
end
function (J::Jacobian{JVP_Partial})(
    res::AbstractVector{R}, v_r::AbstractVector{R}, α::R, β::R
) where {R}
    # [p, B]
    v = reshape(v_r, (length(J.ps), J.size_x[end]))
    J_v_array = map(axes(v, ndims(v))) do b
        # [p]
        v_b = selectdim(v, ndims(v), b)
        # [Lx, Ly, F, T * B]
        J_v_b = getdata(
            Lux.jacobian_vector_product(J.sm_x_r, AutoForwardDiff(), J.ps, v_b),
        )
        # [Lx * Ly * F * T * B]
        J_v_b_r = vec(J_v_b)
        return J_v_b_r
    end
    # [Lx * Ly * F * T * B, B]
    J_v = stack(J_v_array)
    # [Lx * Ly * F * T * B * B]
    J_v_r = vec(J_v)

    if β === zero(β)
        @inbounds res .= α .* J_v_r
    else
        @inbounds res .= α .* J_v_r .+ β .* res
    end

    return res
end
##
function (J::Jacobian{VJP_Serial})(
    res::AbstractVector{R}, v::AbstractVector{R}, α::R, β::R
) where {R}
    # [Lx, Ly, F, T * B]
    (Lx, Ly, F, T, B) = J.size_x
    v_r = reshape(v, (Lx, Ly, F, T * B))
    # [p]
    J_v_r = getdata(
        Lux.vector_jacobian_product(J.sm_x_r, AutoZygote(), J.ps, v_r)
    )

    if β === zero(β)
        @inbounds res .= α .* J_v_r
    else
        @inbounds res .= α .* J_v_r .+ β .* res
    end

    return res
end
function (J::Jacobian{JVP_Serial})(
    res::AbstractVector{R}, v::AbstractVector{R}, α::R, β::R
) where {R}
    # [Lx, Ly, F, T * B]
    J_v = getdata(
        Lux.jacobian_vector_product(J.sm_x_r, AutoForwardDiff(), J.ps, v)
    )
    # [Lx * Ly * F * T * B]
    J_v_r = vec(J_v)

    if β === zero(β)
        @inbounds res .= α .* J_v_r
    else
        @inbounds res .= α .* J_v_r .+ β .* res
    end

    return res
end
##
