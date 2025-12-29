##
@concrete struct Up <:
                 AbstractLuxContainerLayer{(:chain_in, :conv_skip, :chain_out)}
    chain_in
    conv_skip
    chain_out
    K
    chs_in
    chs_out
end
function Up(chs_in::Int; K::Int=3)
    ##
    chs_out = div(chs_in, 4)
    ##
    padding = Tuple(mapfoldl(i -> [cld(i, 2), fld(i, 2)], vcat, (K, K) .- 1))
    padder = WrappedFunction(x -> pad_circular(x, padding))
    ## Chain In
    norm_in = LayerNorm((1, 1, chs_in), swish; dims=(1, 2))
    shuffle = PixelShuffle(2)
    chain_in = Chain(norm_in, shuffle)
    ## Conv Skip
    conv_skip = Conv((1, 1), chs_out => chs_out; init_weight=kaiming_normal)
    ## Chain Out
    conv_in = Conv((K, K), chs_out => chs_out; init_weight=kaiming_normal)
    norm_out = LayerNorm((1, 1, chs_out), swish; dims=(1, 2))
    conv_out = Conv((K, K), chs_out => chs_out; init_weight=kaiming_normal)
    ##
    chain_out = Chain(padder, conv_in, norm_out, padder, conv_out)
    ##
    return Up(chain_in, conv_skip, chain_out, K, chs_in, chs_out)
end
##
function (m::Up)(x::AbstractArray{R,4}, ps, st::NamedTuple) where {R}
    ## Chain In
    x_chain_in, st_chain_in = Lux.apply(m.chain_in, x, ps.chain_in, st.chain_in)
    ## Skip Path
    x_conv_skip, st_conv_skip = Lux.apply(
        m.conv_skip, x_chain_in, ps.conv_skip, st.conv_skip
    )
    ## Chain Out
    x_chain_out, st_chain_out = Lux.apply(
        m.chain_out, x_chain_in, ps.chain_out, st.chain_out
    )
    ## Skip
    m_x = (x_chain_out .+ x_conv_skip) ./ sqrt(R(2))
    ##
    st_ = (chain_in=st_chain_in, conv_skip=st_conv_skip, chain_out=st_chain_out)
    ##
    return m_x, st_
end
##
