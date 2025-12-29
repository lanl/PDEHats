##
@concrete struct Encoder <: AbstractLuxWrapperLayer{:chain}
    chain <: Chain
end

function Encoder(chs::Int, depth::Int, number_heads::Int; r::Int=4)
    ##
    take_first = WrappedFunction(x -> first(x))
    ##
    layers = map(1:depth) do d
        layer = Chain(
            SkipConnection(
                Chain(
                    RMSNorm((1,)),
                    MultiHeadAttention(chs; nheads=number_heads),
                    take_first,
                ),
                +,
            ),
            SkipConnection(
                Chain(
                    RMSNorm((1,)),
                    Chain(Dense(chs => r * chs, swish), Dense(r * chs => chs)),
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
