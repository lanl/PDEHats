##
function loss_mse_scaled(target::AbstractArray, pred::AbstractArray)
    scale = sqrt.(mean(abs2, target; dims=(1, 2))) .+ 1.0f-6
    loss = mean(abs2.(target .- pred) ./ scale)
    return loss
end
##
function loss_mse_scaled(
    model::AbstractLuxLayer, ps, st::NamedTuple, obs::Tuple
)
    (input, target) = obs
    pred, st_ = Lux.apply(model, input, ps, st)
    loss = loss_mse_scaled(model, ps, st_, input, target, pred)
    return loss, st_, (;)
end
function loss_mse_scaled(
    model::AbstractLuxLayer,
    ps,
    st::NamedTuple,
    input::AbstractArray,
    target::AbstractArray,
    pred::AbstractArray,
)
    loss = loss_mse_scaled(target, pred)
    return loss
end
