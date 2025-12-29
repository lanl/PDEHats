##
function get_irreps(m::Array{Float32,2})
    ##
    @assert size(m, 1) == size(m, 2)
    n = size(m, 1)
    u = ones(Float32, n) ./ Float32(sqrt(n))
    @tullio P[i, j] := u[i] * u[j]
    Q = I(n) .- P
    delta(i, j) = i == j ? 1 : 0
    #
    normalizer = norm(m)
    m_normalized = m ./ normalizer
    #
    @tullio m_1[i, j] := P[i, a] * m_normalized[a, b] * P[b, j]
    @tullio m_2[i, j] := P[i, a] * m_normalized[a, b] * Q[b, j]
    @tullio m_3[i, j] := Q[i, a] * m_normalized[a, b] * P[b, j]
    #
    @tullio m_4[i, j] :=
        Q[i, a] * (m_normalized[a, b] - m_normalized[b, a]) * Q[b, j] / 2
    @tullio S[i, j] :=
        Q[i, a] * (m_normalized[a, b] + m_normalized[b, a]) * Q[b, j] / 2
    #
    @tullio m_5[i, j] := S[a, a] * Q[i, j] / tr(Q)
    m_6 = S .- m_5
    ## Check norm
    norm_m_1 = norm(m_1)
    norm_m_2 = norm(m_2)
    norm_m_3 = norm(m_3)
    norm_m_4 = norm(m_4)
    norm_m_5 = norm(m_5)
    norm_m_6 = norm(m_6)
    #
    norm_m = sqrt(
        norm_m_1^2 +
        norm_m_2^2 +
        norm_m_3^2 +
        norm_m_4^2 +
        norm_m_5^2 +
        norm_m_6^2,
    )
    @assert isapprox(norm_m, 1)
    ##
    irreps = [m_1, m_2, m_3, m_4, m_5, m_6]
    # Check orthogonality
    map(Iterators.product(1:6, 1:6)) do (i, j)
        overlap = tr(irreps[i]' * irreps[j])
        if i != j
            @assert overlap < eps(Float32)
        end
        return nothing
    end
    irreps = map(i -> i .* normalizer, irreps)
    ##
    return irreps
end
##
function get_irreps(M::Array{Float32,4})
    ##
    @assert size(M, 1) == size(M, 3)
    T_max = size(M, 1)
    irreps = map(Iterators.product(1:T_max, 1:T_max)) do (t, T)
        m = M[t, :, T, :]
        # [3, 3, 6]
        return stack(get_irreps(m))
    end
    # [3, 3, 6, t, T]
    return stack(irreps)
end
function get_invariants(M::Array{Float32,4})
    ##
    @assert size(M, 1) == size(M, 3)
    T_max = size(M, 1)
    invariants_array = map(Iterators.product(1:T_max, 1:T_max)) do (t, T)
        m = M[t, :, T, :]
        irreps = get_irreps(m)
        invariants = map(i -> norm(i), irreps)
        return invariants
    end
    ## [6, t, T]
    invariants = stack(invariants_array)
    return invariants
end
##
