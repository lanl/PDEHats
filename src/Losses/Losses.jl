##
include("loss_mse.jl")
include("loss_mse_scaled.jl")
##
function loss_mass(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    mass_input = mean(copy(selectdim(input, 3, 1:1)); dims=(1, 2))
    mass_pred = mean(copy(selectdim(pred, 3, 1:1)); dims=(1, 2))
    mass_diff = mass_pred .- mass_input
    normalizer = mass_input .+ 1.0f-6
    loss = abs2.(mass_diff) ./ normalizer
    return mean(loss)
end
##
function loss_mass(model::AbstractLuxLayer, ps, st::NamedTuple, obs::Tuple)
    (input, target) = obs
    pred, st_ = Lux.apply(model, input, ps, st)
    loss = loss_mass(model, ps, st_, input, target, pred)
    return loss, st_, (;)
end
function loss_mass(
    model::AbstractLuxLayer,
    ps,
    st::NamedTuple,
    input::AbstractArray,
    target::AbstractArray,
    pred::AbstractArray,
)
    loss = loss_mass(input, target, pred)
    return loss
end
##
function loss_energy(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    energy_input = mean(copy(selectdim(input, 3, 4:4)); dims=(1, 2))
    energy_pred = mean(copy(selectdim(pred, 3, 4:4)); dims=(1, 2))
    energy_diff = energy_pred .- energy_input
    normalizer = energy_input .+ 1.0f-6
    loss = abs2.(energy_diff) ./ normalizer
    return mean(loss)
end
function loss_energy(model::AbstractLuxLayer, ps, st::NamedTuple, obs::Tuple)
    (input, target) = obs
    pred, st_ = Lux.apply(model, input, ps, st)
    loss = loss_energy(model, ps, st_, input, target, pred)
    return loss, st_, (;)
end
function loss_energy(
    model::AbstractLuxLayer,
    ps,
    st::NamedTuple,
    input::AbstractArray,
    target::AbstractArray,
    pred::AbstractArray,
)
    loss = loss_energy(input, target, pred)
    return loss
end
