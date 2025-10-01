##
include("UNet/UNet.jl")
include("ViT/ViT.jl")
## Utility
function get_model(rng::AbstractRNG, chs::Int, name_model::Symbol; L::Int=128)
    if name_model == :UNet
        model = UNet(chs)
    elseif name_model == :ViT
        model = ViT(chs; L=L)
    else
        throw("only name_model == :UNet and :ViT are supported")
    end
    ps, st = Lux.setup(Lux.replicate(rng), model)
    st = Lux.trainmode(st)
    return model, ps, st
end
function get_model(chs::Int, name_model::Symbol; L::Int=128)
    if name_model == :UNet
        model = UNet(chs)
    elseif name_model == :ViT
        model = ViT(chs; L=L)
    else
        throw("only name_model == :UNet and :ViT are supported")
    end
    return model
end
