##
include("attn.jl")
include("up.jl")
include("down.jl")
##
@concrete struct UNet <: AbstractLuxContainerLayer{(
    :proj_in,
    :down_1,
    :down_2,
    :down_3,
    :down_4,
    :across,
    :up_4,
    :up_3,
    :up_2,
    :up_1,
    :proj_out,
)}
    proj_in
    down_1
    down_2
    down_3
    down_4
    across
    up_4
    up_3
    up_2
    up_1
    proj_out
    chs
end
##
function UNet(chs::Int; r::Int=2, F::Int=4)
    ## Mass, Momentum, Energy
    N_fields = F
    ## Space
    K = 11
    K_1 = 9
    K_2 = 7
    K_3 = 5
    K_4 = 3
    ##
    padding = get_padding_circular(K)
    padder = WrappedFunction(x -> pad_circular(x, padding))
    padding_4 = get_padding_circular(K_4)
    padder_4 = WrappedFunction(x -> pad_circular(x, padding_4))
    ##
    proj_in = Chain(
        padder, Conv((K, K), N_fields => chs; init_weight=kaiming_normal)
    )
    ##
    down_1 = Down(chs; K=K_1)
    down_2 = Down(2 * chs; K=K_2)
    down_3 = Down(4 * chs; K=K_3)
    down_4 = Down(8 * chs; K=K_4)
    ##
    across = Chain(
        Chain(RMSNorm((1, 1)), swish),
        padder_4,
        Conv((K_4, K_4), 16 * chs => 32 * chs; init_weight=kaiming_normal),
        AttnBlock(32 * chs; r=r),
    )
    ##
    up_4 = Up(32 * chs; K=K_4)
    up_3 = Up(16 * chs; K=K_3)
    up_2 = Up(8 * chs; K=K_2)
    up_1 = Up(4 * chs; K=K_1)
    ##
    proj_out = Chain(
        Chain(RMSNorm((1, 1)), swish),
        padder,
        Conv((K, K), 2 * chs => 2 * chs; init_weight=kaiming_normal),
        Chain(RMSNorm((1, 1)), swish),
        padder,
        Conv((K, K), 2 * chs => 2 * chs; init_weight=kaiming_normal),
        Conv((1, 1), 2 * chs => N_fields; init_weight=kaiming_normal),
    )
    ##
    return UNet(
        proj_in,
        down_1,
        down_2,
        down_3,
        down_4,
        across,
        up_4,
        up_3,
        up_2,
        up_1,
        proj_out,
        chs,
    )
end
##
function (m::UNet)(x::AbstractArray{R,5}, ps, st::NamedTuple) where {R}
    (Lx, Ly, F, T, B) = size(x)
    x_r = reshape(x, (Lx, Ly, F, T * B))
    m_x_r, st_ = Lux.apply(m, x_r, ps, st)
    m_x = reshape(m_x_r, size(x))
    return m_x, st_
end
function (m::UNet)(x::AbstractArray{R,4}, ps, st::NamedTuple) where {R}
    ## In
    x_proj_in, st_proj_in = Lux.apply(m.proj_in, x, ps.proj_in, st.proj_in)
    ## Down
    x_down_1, st_down_1 = Lux.apply(m.down_1, x_proj_in, ps.down_1, st.down_1)
    x_down_2, st_down_2 = Lux.apply(m.down_2, x_down_1, ps.down_2, st.down_2)
    x_down_3, st_down_3 = Lux.apply(m.down_3, x_down_2, ps.down_3, st.down_3)
    x_down_4, st_down_4 = Lux.apply(m.down_4, x_down_3, ps.down_4, st.down_4)
    ## 4
    x_4_across, st_across = Lux.apply(m.across, x_down_4, ps.across, st.across)
    x_up_4, st_up_4 = Lux.apply(m.up_4, x_4_across, ps.up_4, st.up_4)
    ## 3
    x_3 = cat(x_up_4, x_down_3; dims=3)
    x_up_3, st_up_3 = Lux.apply(m.up_3, x_3, ps.up_3, st.up_3)
    ## 2
    x_2 = cat(x_up_3, x_down_2; dims=3)
    x_up_2, st_up_2 = Lux.apply(m.up_2, x_2, ps.up_2, st.up_2)
    ## 1
    x_1 = cat(x_up_2, x_down_1; dims=3)
    x_up_1, st_up_1 = Lux.apply(m.up_1, x_1, ps.up_1, st.up_1)
    ## Out
    x_0 = cat(x_up_1, x_proj_in; dims=3)
    m_x, st_proj_out = Lux.apply(m.proj_out, x_0, ps.proj_out, st.proj_out)
    ##
    st_ = (
        proj_in=st_proj_in,
        down_1=st_down_1,
        down_2=st_down_2,
        down_3=st_down_3,
        down_4=st_down_4,
        across=st_across,
        up_4=st_up_4,
        up_3=st_up_3,
        up_2=st_up_2,
        up_1=st_up_1,
        proj_out=st_proj_out,
    )
    ##
    return m_x, st_
end
##
