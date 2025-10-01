## GLOBAL CONSERVATION
function get_Q_global_flucts(trajectories::AbstractArray{R,5}) where {R}
    ##
    @assert size(trajectories, 3) == 4
    ##
    (Lx, Ly, F, T, B) = size(trajectories)
    ##
    Q_global = sum(trajectories; dims=(1, 2))
    density_global = selectdim(Q_global, 3, 1:1)
    momentum_global = selectdim(Q_global, 3, 2:3)
    energy_global = selectdim(Q_global, 3, 4:4)
    ## Density
    density_global_fwd = selectdim(density_global, 4, 2:T)
    density_global_ref = selectdim(density_global, 4, 1:(T - 1))
    density_global_diffs = density_global_fwd .- density_global_ref
    normalizer_density = (density_global_ref .+ 1.0f-6) ./ 100.0f0
    density_flucts = (density_global_diffs ./ normalizer_density)
    ## Energy
    energy_global_fwd = selectdim(energy_global, 4, 2:T)
    energy_global_ref = selectdim(energy_global, 4, 1:(T - 1))
    energy_global_diffs = energy_global_fwd .- energy_global_ref
    normalizer_energy = (energy_global_ref .+ 1.0f-6) ./ 100.0f0
    energy_flucts = (energy_global_diffs ./ normalizer_energy)
    ## Momentum
    momentum = selectdim(trajectories, 3, 2:3)
    momentum_global_fwd = selectdim(momentum_global, 4, 2:T)
    momentum_global_ref = selectdim(momentum_global, 4, 1:(T - 1))
    momentum_global_diffs = momentum_global_fwd .- momentum_global_ref
    momentum_ref = selectdim(momentum, 4, 1:(T - 1))
    momentum_rms = Lx .* Ly .* sqrt.(mean(abs2.(momentum_ref); dims=(1, 2, 3)))
    normalizer_momentum = (momentum_rms .+ 1.0f-6) ./ 100.0f0
    momentum_flucts = (momentum_global_diffs ./ normalizer_momentum)
    #
    Q_flucts_normed = cat(
        density_flucts, momentum_flucts, energy_flucts; dims=3
    )
    return Q_flucts_normed
end
function get_Q_global_flucts(
    input::AbstractArray{R,5}, pred::AbstractArray{R,5}
) where {R}
    ##
    @assert size(pred, 3) == 4
    @assert size(input, 3) == 4
    ##
    (Lx, Ly, F, T, B) = size(pred)
    ##
    Q_global_input = sum(input; dims=(1, 2))
    Q_global_pred = sum(pred; dims=(1, 2))
    ## Density
    density_global_input = selectdim(Q_global_input, 3, 1:1)
    density_global_pred = selectdim(Q_global_pred, 3, 1:1)
    density_global_diffs = density_global_pred .- density_global_input
    normalizer_density = (density_global_input .+ 1.0f-6) ./ 100.0f0
    density_flucts = (density_global_diffs ./ normalizer_density)
    ## Energy
    energy_global_input = selectdim(Q_global_input, 3, 4:4)
    energy_global_pred = selectdim(Q_global_pred, 3, 4:4)
    energy_global_diffs = energy_global_pred .- energy_global_input
    normalizer_energy = (energy_global_input .+ 1.0f-6) ./ 100.0f0
    energy_flucts = (energy_global_diffs ./ normalizer_energy)
    ## Momentum
    momentum_input = selectdim(input, 3, 2:3)
    momentum_global_pred = selectdim(Q_global_pred, 3, 2:3)
    momentum_global_input = selectdim(Q_global_input, 3, 2:3)
    momentum_global_diffs = momentum_global_pred .- momentum_global_input
    momentum_input_rms =
        Lx .* Ly .* sqrt.(mean(abs2.(momentum_input); dims=(1, 2, 3)))
    normalizer_momentum = (momentum_input_rms .+ 1.0f-6) ./ 100.0f0
    momentum_flucts = (momentum_global_diffs ./ normalizer_momentum)
    ##
    Q_flucts_normed = cat(
        density_flucts, momentum_flucts, energy_flucts; dims=3
    )
    return Q_flucts_normed
end
##
function equation_of_state_energy(
    density::AbstractArray{R},
    velocity::AbstractArray{R},
    pressure::AbstractArray{R};
    gamma::Float32=1.4f0,
) where {R}
    # https://arxiv.org/abs/2405.19101
    energy_eos =
        (0.5f0) .* density .* sum(abs2.(velocity); dims=3) .+
        (pressure ./ (gamma - 1))
    return energy_eos
end
##
function equation_of_state_pressure(
    density::AbstractArray{R},
    velocity::AbstractArray{R},
    energy::AbstractArray{R};
    gamma::Float32=1.4f0,
) where {R}
    # https://arxiv.org/abs/2405.19101
    pressure_eos =
        (gamma - 1) .*
        (energy .- (0.5f0) .* density .* sum(abs2.(velocity); dims=3))
    return pressure_eos
end
##
function get_equation_of_state_gamma(data::AbstractArray{R}) where {R}
    @assert size(data, 3) == 5
    ##
    density = selectdim(data, 3, 1:1)
    velocity = selectdim(data, 3, 2:3)
    pressure = selectdim(data, 3, 4:4)
    energy = selectdim(data, 3, 5:5)
    kinetic = 0.5f0 .* density .* sum(abs2.(velocity); dims=3)
    gamma_eos = @. 1 + pressure / ((energy - kinetic) + 1.0f-6)
    return gamma_eos
end
##
function get_Q_local_flucts(trajectories::AbstractArray{Float32,5})
    ##
    (Lx, Ly, F, T, B) = size(trajectories)
    ##
    density = copy(selectdim(trajectories, 3, 1:1))
    momenta = copy(selectdim(trajectories, 3, 2:3))
    energy = copy(selectdim(trajectories, 3, 4:4))
    ##
    velocity = momenta ./ (density .+ 1.0f-6)
    #
    pressure = equation_of_state_pressure(density, velocity, energy)
    ## Mass conservation
    dt_density = diff_time(density)
    div_momenta = div_space(momenta)
    ## Energy conservation
    grad_pressure = grad_space(pressure)
    grad_energy = grad_space(energy)
    div_velocity = div_space(velocity)
    div_energy_current_1 = div_velocity .* (energy .+ pressure)
    div_energy_current_2 = sum(
        velocity .* (grad_energy .+ grad_pressure); dims=3
    )
    div_energy_current = div_energy_current_1 .+ div_energy_current_2
    dt_energy = diff_time(energy)
    ## Momentum conservation
    δ = [1 0; 0 1]
    @tullio stress[x, y, i, j, t, b] := (
        δ[i, j] * pressure[x, y, 1, t, b] +
        density[x, y, 1, t, b] *
        velocity[x, y, i, t, b] *
        velocity[x, y, j, t, b]
    )
    div_stress = div_space(stress)
    dt_momenta = diff_time(momenta)
    ##
    dt_rho = cat(dt_density, dt_momenta, dt_energy; dims=3)
    _div_j = cat(div_momenta, div_stress, div_energy_current; dims=3)
    div_j = selectdim(_div_j, 4, 1:(T - 1))
    source_rho = dt_rho .+ div_j
    ##
    normalizer_density = (sum(density; dims=(1, 2)) .+ 1.0f-6)
    normalizer_energy = (sum(energy; dims=(1, 2)) .+ 1.0f-6)
    normalizer_momenta = (
        Lx * Ly * sqrt.(mean(abs2.(momenta); dims=(1, 2, 3))) .+ 1.0f-6
    )
    _normalizer =
        cat(
            normalizer_density,
            normalizer_momenta,
            normalizer_momenta,
            normalizer_energy;
            dims=3,
        ) ./ 100.0f0
    normalizer = selectdim(_normalizer, 4, 1:(T - 1))
    ##
    dt_rho_normalized = dt_rho ./ normalizer
    div_j_normalized = div_j ./ normalizer
    source_rho_normalized = source_rho ./ normalizer
    ##
    return dt_rho_normalized, div_j_normalized, source_rho_normalized
end
