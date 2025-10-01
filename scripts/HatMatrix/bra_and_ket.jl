## Bras
function bra_Q_mass(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    delta_mass = dropdims(
        sum(copy(selectdim(pred_delta, 3, 1:1)); dims=(1, 2)); dims=(1, 2, 3)
    )
    pred_mass = dropdims(
        sum(copy(selectdim(pred, 3, 1:1)); dims=(1, 2)); dims=(1, 2, 3)
    )
    return delta_mass ./ (pred_mass .+ 1.0f-6)
end
function bra_Q_energy(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    delta_energy = dropdims(
        sum(copy(selectdim(pred_delta, 3, 4:4)); dims=(1, 2)); dims=(1, 2, 3)
    )
    pred_energy = dropdims(
        sum(copy(selectdim(pred, 3, 4:4)); dims=(1, 2)); dims=(1, 2, 3)
    )
    return delta_energy ./ (pred_energy .+ 1.0f-6)
end
function bra_C_mass(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    r = get_r_mass(input, target, pred)
    response = -1.0f0 .* sum(pred_delta .* r; dims=(1, 2, 3))
    return dropdims(response; dims=(1, 2, 3))
end
function bra_C_momentum_x(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    r = get_r_momentum_x(input, target, pred)
    response = -1.0f0 .* sum(pred_delta .* r; dims=(1, 2, 3))
    return dropdims(response; dims=(1, 2, 3))
end
function bra_C_momentum_y(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    r = get_r_momentum_y(input, target, pred)
    response = -1.0f0 .* sum(pred_delta .* r; dims=(1, 2, 3))
    return dropdims(response; dims=(1, 2, 3))
end
function bra_C_energy(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    r = get_r_energy(input, target, pred)
    response = -1.0f0 .* sum(pred_delta .* r; dims=(1, 2, 3))
    return dropdims(response; dims=(1, 2, 3))
end
function bra_C_smse(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    r = get_r_smse(input, target, pred)
    response = -1.0f0 .* sum(pred_delta .* r; dims=(1, 2, 3))
    return dropdims(response; dims=(1, 2, 3))
end
## Kets
function ket_C_smse(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    r = get_r_smse(input, target, pred)
    return r
end
function ket_C_mass(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    r = get_r_mass(input, target, pred)
    return r
end
function ket_Q_mass(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    target_mass = copy(selectdim(target, 3, 1:1))
    delta_mass = ones_like(target_mass)
    delta_target = zeros_like(target)
    delta_target[:, :, 1:1, :, :] .= delta_mass
    return delta_target
end
## Utils
function get_r_smse(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    r_mse = pred .- target
    scale = sqrt.(mean(abs2.(target); dims=(1, 2))) .+ 1.0f-6
    r_smse = r_mse ./ scale
    return r_smse
end
function get_r_mass(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    mass_input = sum(copy(selectdim(input, 3, 1:1)); dims=(1, 2))
    mass_pred = sum(copy(selectdim(pred, 3, 1:1)); dims=(1, 2))
    mass_diff = mass_pred .- mass_input
    normalizer = mass_input .+ 1.0f-6
    r_mass = zeros_like(input)
    r_mass[:, :, 1:1, :, :] .= mass_diff ./ normalizer
    return r_mass
end
function get_r_momentum_x(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    momentum_x_input = sum(copy(selectdim(input, 3, 2:2)); dims=(1, 2))
    momentum_x_pred = sum(copy(selectdim(pred, 3, 2:2)); dims=(1, 2))
    momentum_x_diff = momentum_x_pred .- momentum_x_input
    ##
    momentum_input = copy(selectdim(input, 3, 2:3))
    (Lx, Ly, F, T, B) = size(input)
    momentum_rms =
        Lx .* Ly .* sqrt.(mean(abs2.(momentum_input); dims=(1, 2, 3)))
    normalizer = momentum_rms .+ 1.0f-6
    ##
    r_momentum_x = zeros_like(input)
    r_momentum_x[:, :, 4:4, :, :] .= momentum_x_diff ./ normalizer
    return r_momentum_x
end
function get_r_momentum_y(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    momentum_y_input = sum(copy(selectdim(input, 3, 3:3)); dims=(1, 2))
    momentum_y_pred = sum(copy(selectdim(pred, 3, 3:3)); dims=(1, 2))
    momentum_y_diff = momentum_y_pred .- momentum_y_input
    ##
    momentum_input = copy(selectdim(input, 3, 2:3))
    (Lx, Ly, F, T, B) = size(input)
    momentum_rms =
        Lx .* Ly .* sqrt.(mean(abs2.(momentum_input); dims=(1, 2, 3)))
    normalizer = momentum_rms .+ 1.0f-6
    ##
    r_momentum_y = zeros_like(input)
    r_momentum_y[:, :, 4:4, :, :] .= momentum_y_diff ./ normalizer
    return r_momentum_y
end
function get_r_energy(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
)
    energy_input = sum(copy(selectdim(input, 3, 4:4)); dims=(1, 2))
    energy_pred = sum(copy(selectdim(pred, 3, 4:4)); dims=(1, 2))
    energy_diff = energy_pred .- energy_input
    normalizer = energy_input .+ 1.0f-6
    r_energy = zeros_like(input)
    r_energy[:, :, 4:4, :, :] .= energy_diff ./ normalizer
    return r_energy
end
