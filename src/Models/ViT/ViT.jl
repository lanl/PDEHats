## Inspired by Lux.jl implementation
include("embed.jl")
include("encoder.jl")
##
@concrete struct ViT <: Lux.AbstractLuxWrapperLayer{:chain}
    chain <: Lux.Chain
    chs <: Int
end

function ViT(
    chs::Int;
    K::Int=5,
    depth::Int=8,
    N_heads::Int=8,
    r::Int=2,
    N_fields::Int=4,
    L::Int=128,
)
    ##
    k = K - 1
    l = div(L, k)
    ##
    patch_embed = PatchEmbed(N_fields => chs; K=K)
    pos_embed = PosEmbedding(chs, l * l)
    encoder = Encoder(chs, depth, N_heads; r=r)
    layer_norm = LayerNorm((chs, 1); dims=1, affine=true)
    dense = Dense(chs => k * k * N_fields)
    rshp_1 = ReshapeLayer((k * k, N_fields, l * l))
    permuter = WrappedFunction(x -> permutedims(x, (1, 3, 2, 4)))
    rshp_2 = ReshapeLayer((L, L, N_fields))
    chain = Chain(
        patch_embed,
        pos_embed,
        encoder,
        layer_norm,
        dense,
        rshp_1,
        permuter,
        rshp_2,
    )
    ##
    return ViT(chain, chs)
end
##
function (m::ViT)(x::AbstractArray{Float32,5}, ps, st::NamedTuple)
    (Lx, Ly, F, T, B) = size(x)
    x_r = reshape(x, (Lx, Ly, F, T * B))
    m_x_r, st_ = Lux.apply(m, x_r, ps, st)
    m_x = reshape(m_x_r, size(x))
    return m_x, st_
end
## Rollout
function rollout(m::ViT, x::AbstractArray{R,5}, ps, st::NamedTuple;) where {R}
    T = size(x, 4)
    q = selectdim(x, 4, 1:1)
    preds = map(1:T) do t
        q, st = Lux.apply(m, q, ps, st)
        return q
    end
    trajectory = cat(preds...; dims=4)
    return trajectory, st
end
