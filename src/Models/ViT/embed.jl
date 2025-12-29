##
@concrete struct PatchEmbed <: AbstractLuxContainerLayer{(:padder, :patcher)}
    padder <: AbstractLuxLayer
    patcher <: AbstractLuxLayer
end

function (m::PatchEmbed)(x::AbstractArray{Float32,4}, ps, st::NamedTuple)
    #
    (Lx, Ly, chs_in, B) = size(x)
    # [2+Lx+2, 2+Ly+2, chs_in, B]
    x_padder, st_padder = Lux.apply(m.padder, x, ps.padder, st.padder)
    # [lx, ly, chs_out, B]
    x_patcher, st_patcher = Lux.apply(
        m.patcher, x_padder, ps.patcher, st.patcher
    )
    (lx, ly, chs_out, B) = size(x_patcher)
    # [lx * ly, chs_out, B]
    x_patcher_r = reshape(x_patcher, (lx * ly, chs_out, B))
    # [chs_out, lx * ly, B]
    x_patcher_r_p = permutedims(x_patcher_r, (2, 1, 3))
    #
    m_x = x_patcher_r_p
    st_ = (padder=st_padder, patcher=st_patcher)
    #
    return m_x, st_
end
function PatchEmbed(P::Pair; K::Int=5, stride::Int=4)
    padding = get_padding_circular(K)
    padder = WrappedFunction(x -> pad_circular(x, padding))
    patcher = Conv((K, K), P; stride=stride)
    return PatchEmbed(padder, patcher)
end
##
@concrete struct PosEmbedding <: AbstractLuxLayer
    chs::Int
    number_patches::Int
end
function LuxCore.initialparameters(rng::AbstractRNG, m::PosEmbedding)
    return (; embeddings=randn(Float32, m.chs, m.number_patches))
end
(m::PosEmbedding)(x::AbstractArray, ps, st::NamedTuple) = x .+ ps.embeddings, st
