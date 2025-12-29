##
@concrete struct AttnBlock <: AbstractLuxContainerLayer{(:norm_in, :mha, :ffn)}
    norm_in
    mha
    ffn
end
function AttnBlock(chs::Int; N_heads::Int=4, r::Int=4)
    ##
    norm_in = RMSNorm((1, 1))
    mha = MultiHeadAttention(chs; nheads=N_heads)
    ffn = Chain(
        RMSNorm((1, 1)),
        Conv((1, 1), chs => r * chs, swish; init_weight=kaiming_normal),
        Conv((1, 1), r * chs => chs; init_weight=kaiming_normal),
    )
    ##
    return AttnBlock(norm_in, mha, ffn)
end
##
function (m::AttnBlock)(x::AbstractArray{R,4}, ps, st::NamedTuple) where {R}
    ##
    (Lx, Ly, F, TB) = size(x)
    x_norm_in, st_norm_in = Lux.apply(m.norm_in, x, ps.norm_in, st.norm_in)
    # [Lx * Ly, F, TB]
    x_norm_in_r = reshape(x_norm_in, (Lx * Ly, F, TB))
    # [F, Lx * Ly, TB]
    x_norm_in_r_p = permutedims(x_norm_in_r, (2, 1, 3))
    (x_mha_r_p, _), st_mha = Lux.apply(m.mha, x_norm_in_r_p, ps.mha, st.mha)
    # [Lx * Ly, F, TB]
    x_mha_r = permutedims(x_mha_r_p, (2, 1, 3))
    # [Lx * Ly, F, TB]
    x_mha = reshape(x_mha_r, size(x))
    ##
    x_mha_skip = x_mha .+ x
    x_ffn, st_ffn = Lux.apply(m.ffn, x_mha_skip, ps.ffn, st.ffn)
    ##
    m_x = (x_ffn + x_mha_skip) ./ sqrt(R(2))
    st_ = (; norm_in=st_norm_in, mha=st_mha, ffn=st_ffn)
    ##
    return m_x, st_
end
##
