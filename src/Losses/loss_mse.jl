##
function loss_mse(target::AbstractArray, pred::AbstractArray)
    loss = mean(abs2.(target .- pred))
    return loss
end
function loss_mse(model::AbstractLuxLayer, ps, st::NamedTuple, obs::Tuple)
    (input, target) = obs
    pred, st_ = Lux.apply(model, input, ps, st)
    loss = loss_mse(model, ps, st_, input, target, pred)
    return loss, st_, (;)
end
function loss_mse(
    model::AbstractLuxLayer,
    ps,
    st::NamedTuple,
    input::AbstractArray,
    target::AbstractArray,
    pred::AbstractArray,
)
    loss = loss_mse(target, pred)
    return loss
end
