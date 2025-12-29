## Bras
function bra_C_mass(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    r = get_r_mass(input, target, pred)
    response = dropdims(mean(pred_delta .* r; dims=(1, 2, 3)); dims=(1, 2, 3))
    return response
end
function bra_C_energy(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    r = get_r_energy(input, target, pred)
    response = dropdims(mean(pred_delta .* r; dims=(1, 2, 3)); dims=(1, 2, 3))
    return response
end
function bra_C_smse(
    input::AbstractArray{Float32,5},
    target::AbstractArray{Float32,5},
    pred::AbstractArray{Float32,5},
    pred_delta::AbstractArray{Float32,5},
)
    r = get_r_smse(input, target, pred)
    response = dropdims(mean(pred_delta .* r; dims=(1, 2, 3)); dims=(1, 2, 3))
    return response
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
##
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
