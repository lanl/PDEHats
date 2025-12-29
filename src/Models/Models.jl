##
include("UNet/UNet.jl")
include("ViT/ViT.jl")
include("FNO/FNO.jl")
## Utility
function get_model(
    rng::AbstractRNG, chs::Int, name_model::Symbol, name_data::Symbol
)
    L, F = get_L_and_F(name_data)
    if name_model == :UNet
        model = UNet(chs; F=F)
    elseif name_model == :ViT
        model = ViT(chs; F=F, L=L)
    else
        throw("only name_model == :UNet and :ViT are supported")
    end
    ps, st = Lux.setup(Lux.replicate(rng), model)
    st = Lux.trainmode(st)
    return model, ps, st
end
function get_model(chs::Int, name_model::Symbol, name_data::Symbol)
    L, F = get_L_and_F(name_data)
    if name_model == :UNet
        model = UNet(chs; F=F)
    elseif name_model == :ViT
        model = ViT(chs; F=F, L=L)
    else
        throw("only name_model == :UNet and :ViT are supported")
    end
    return model
end
function get_L_and_F(name_data::Symbol)
    if name_data == :CE_TEST
        L = 32
        F = 4
    elseif name_data == :NS_TEST
        L = 32
        F = 2
    elseif name_data == :CE
        L = 128
        F = 4
    elseif name_data == :NS
        L = 128
        F = 2
    else
        throw("unsupported name_data")
    end
    return L, F
end
##
