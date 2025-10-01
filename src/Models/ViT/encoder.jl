##
@concrete struct Encoder <: AbstractLuxWrapperLayer{:chain}
    chain <: Lux.Chain
end

function Encoder(chs::Int, depth::Int, number_heads::Int; r::Int=4)
    ##
    take_first = WrappedFunction(x -> first(x))
    ##
    layers = map(1:depth) do d
        layer = Lux.Chain(
            Lux.SkipConnection(
                Lux.Chain(
                    Lux.LayerNorm((chs, 1); dims=1, affine=true),
                    MultiHeadAttention(chs; nheads=number_heads),
                    take_first,
                ),
                +,
            ),
            Lux.SkipConnection(
                Lux.Chain(
                    Lux.LayerNorm((chs, 1); dims=1, affine=true),
                    Lux.Chain(
                        Lux.Dense(chs => r * chs, swish),
                        Lux.Dense(r * chs => chs),
                    ),
                ),
                +,
            ),
        )
        return layer
    end
    chain = Chain(layers...)
    ##
    return Encoder(chain)
end
##
