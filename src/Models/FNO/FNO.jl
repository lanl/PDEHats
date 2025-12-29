##
include("transform.jl")
include("layers.jl")
include("utils.jl")
##
@concrete struct FourierNeuralOperator <: AbstractLuxWrapperLayer{:model}
    model <: AbstractLuxLayer
end

function FourierNeuralOperator(
    modes::NTuple{2,Int},
    in_channels::Integer,
    out_channels::Integer,
    hidden_channels::Integer;
    num_layers::Integer=4,
    lifting_channel_ratio::Integer=2,
    projection_channel_ratio::Integer=2,
    activation=gelu,
    use_channel_mlp::Bool=true,
    channel_mlp_expansion::Real=0.5,
    channel_mlp_skip::Symbol=:soft_gating,
    fno_skip::Symbol=:linear,
    stabilizer=tanh,
)
    lifting_channels = hidden_channels * lifting_channel_ratio
    projection_channels = out_channels * projection_channel_ratio
    positional_embedding = GridEmbedding([(0.0f0, 1.0f0), (0.0f0, 1.0f0)])
    lifting = Chain(
        _Conv((1, 1), in_channels => lifting_channels, activation),
        Conv((1, 1), lifting_channels => hidden_channels),
    )

    projection = Chain(
        _Conv((1, 1), hidden_channels => projection_channels, activation),
        Conv((1, 1), projection_channels => out_channels),
    )

    fno_blocks = Chain(
        [
            SpectralKernel(
                hidden_channels => hidden_channels,
                modes,
                activation;
                stabilizer,
                use_channel_mlp,
                channel_mlp_expansion,
                channel_mlp_skip,
                fno_skip,
            ) for _ in 1:num_layers
        ]...,
    )

    return FourierNeuralOperator(Chain(; lifting, fno_blocks, projection))
end
function FNO(
    modes::NTuple{2,Int},
    in_channels::Integer,
    out_channels::Integer,
    hidden_channels::Integer;
    num_layers::Integer=4,
    lifting_channel_ratio::Integer=2,
    projection_channel_ratio::Integer=2,
    activation=gelu,
    use_channel_mlp::Bool=true,
    channel_mlp_expansion::Real=0.5,
    channel_mlp_skip::Symbol=:soft_gating,
    fno_skip::Symbol=:linear,
    stabilizer=tanh,
)
    lifting_channels = hidden_channels * lifting_channel_ratio
    projection_channels = out_channels * projection_channel_ratio
    grid_boundaries = [(0.0f0, 1.0f0), (0.0f0, 1.0f0)]

    lifting = Chain(
        Conv((1, 1), in_channels => lifting_channels, activation),
        Conv((1, 1), lifting_channels => hidden_channels),
    )

    projection = Chain(
        Conv((1, 1), hidden_channels => projection_channels, activation),
        Conv((1, 1), projection_channels => out_channels),
    )

    return FourierNeuralOperator(Chain(; lifting))
end
##
@concrete struct _Conv <: AbstractLuxWrapperLayer{:layer}
    layer
end
function _Conv(args...; kwargs...)
    return _Conv(Conv(args...; kwargs...))
end
function LuxCore.initialparameters(rng::AbstractRNG, m::_Conv)
    return LuxCore.initialparameters(rng, m.layer)
end
function LuxCore.initialstates(rng::AbstractRNG, m::_Conv)
    return LuxCore.initialstates(rng, m.layer)
end
function (m::_Conv)(x, ps, st)
    ps_real = ComponentArray(; weight=real.(ps.weight), bias=real.(ps.bias))
    return m.layer(x, ps_real, st)
end
