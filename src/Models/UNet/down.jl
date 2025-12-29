##
@concrete struct Down <:
                 AbstractLuxContainerLayer{(:norm_in, :chain_skip, :chain_out)}
    norm_in
    chain_skip
    chain_out
    K
    chs_in
    chs_out
end
function Down(chs_in::Int; K::Int=3)
    ##
    chs_out = 2 * chs_in
    P = chs_in => chs_out
    ##
    padding = Tuple(mapfoldl(i -> [cld(i, 2), fld(i, 2)], vcat, (K, K) .- 1))
    padder = WrappedFunction(x -> pad_circular(x, padding))
    ## Norm In
    norm_in = LayerNorm((1, 1, chs_in), swish; dims=(1, 2))
    ## Chain Skip
    conv_skip = Conv((1, 1), P; init_weight=kaiming_normal)
    chain_skip = Chain(conv_skip, padder)
    ## Chain Out
    conv_in = Conv((K, K), chs_in => chs_out; init_weight=kaiming_normal)
    norm_out = LayerNorm((1, 1, chs_out), swish; dims=(1, 2))
    conv_out = Conv(
        (K, K), chs_out => chs_out; stride=2, init_weight=kaiming_normal
    )
    chain_out = Chain(padder, conv_in, norm_out, padder, conv_out)
    ##
    return Down(norm_in, chain_skip, chain_out, K, chs_in, chs_out)
end
function Lux.initialstates(rng::AbstractRNG, m::Down)
    st_norm_in = Lux.initialstates(Lux.Utils.sample_replicate(rng), m.norm_in)
    st_chain_skip = Lux.initialstates(
        Lux.Utils.sample_replicate(rng), m.chain_skip
    )
    st_chain_out = Lux.initialstates(
        Lux.Utils.sample_replicate(rng), m.chain_out
    )
    st_weights_pool = ones(Float32, (m.K, m.K, 1, m.chs_out)) ./ (m.K^2)
    st = (
        norm_in=st_norm_in,
        chain_skip=st_chain_skip,
        chain_out=st_chain_out,
        weights_pool=st_weights_pool,
    )
    return st
end
##
function (m::Down)(x::AbstractArray{R,4}, ps, st::NamedTuple) where {R}
    ## Norm In
    x_norm_in, st_norm_in = Lux.apply(m.norm_in, x, ps.norm_in, st.norm_in)
    ## Skip Path
    x_chain_skip, st_chain_skip = Lux.apply(
        m.chain_skip, x_norm_in, ps.chain_skip, st.chain_skip
    )
    x_pooled = conv(x_chain_skip, st.weights_pool; groups=m.chs_out, stride=2)
    ## Main Path
    x_chain_out, st_chain_out = Lux.apply(
        m.chain_out, x_norm_in, ps.chain_out, st.chain_out
    )
    ## Skip
    m_x = (x_chain_out .+ x_pooled) ./ sqrt(R(2))
    ##
    st_ = (
        norm_in=st_norm_in,
        chain_skip=st_chain_skip,
        chain_out=st_chain_out,
        weights_pool=st.weights_pool,
    )
    ##
    return m_x, st_
end
##
