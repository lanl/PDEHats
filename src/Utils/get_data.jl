##
function get_data(
    rng::AbstractRNG, name_data::String; T_max::Int=21, p::AbstractFloat=1.0f0
)
    if name_data == "testing"
        ## Testing Slice [32, 32, 5, 7, 4]
        data = load(datadir("sim_test/data_test.jld2"))["data_test"]
        T_max = size(data, 4)
        p = 1.0f0
    else
        ## Full Dataset [128, 128, 5, 21, N]
        data_full = _get_data(name_data)
        N_tot = size(data_full, 5)
        N_keep = round(Int, p * N_tot)
        idx_keep = rand(rng, 1:N_tot, N_keep)
        data = data_full[:, :, :, 1:T_max, idx_keep]
    end
    ##
    return data
end
function get_data(name_data::String; T_max::Int=21, p::AbstractFloat=1.0f0)
    rng = Xoshiro(0)
    return get_data(rng, name_data; T_max=T_max, p=p)
end
##
function _get_data(name_data::String)
    filepath = datadir("sim_pro/CE-$(name_data).nc")
    data = NCDataset(filepath, "r") do ds
        if occursin("RM", name_data)
            data_rm = ds["solution"][:, :, :, :, :]
            data = intercept_data_rm(data_rm)
        else
            data = ds["data"][:, :, :, :, :]
        end
        return data
    end # ds is closed
    return data
end
## Replace tracer with energy
function intercept_data_rm(data::Array)
    #
    density = selectdim(data, 3, 1:1)
    velocity = selectdim(data, 3, 2:3)
    pressure = selectdim(data, 3, 4:4)
    energy = equation_of_state_energy(density, velocity, pressure)
    data_with_energy = cat(density, velocity, pressure, energy; dims=3)
    return data_with_energy
end
## Switch from (M, Vx, Vy, P, E) to (M, Px, Py, E)
function get_trajectories(
    rng::AbstractRNG, name_data::String; T_max::Int=21, p::AbstractFloat=1.0f0
)
    data = get_data(rng, name_data; T_max=T_max, p=p)
    return get_trajectories!(data)
end
function get_trajectories(
    name_data::String; T_max::Int=21, p::AbstractFloat=1.0f0
)
    rng = Xoshiro(0)
    data = get_data(rng, name_data; T_max=T_max, p=p)
    trajectories = get_trajectories(data)
    return trajectories
end
function get_trajectories!(data::AbstractArray{Float32,5})
    @views begin
        # Momentum (Horizontal)
        broadcast!(
            *, data[:, :, 2, :, :], data[:, :, 1, :, :], data[:, :, 2, :, :]
        )
        # Momentum (Vertical)
        broadcast!(
            *, data[:, :, 3, :, :], data[:, :, 1, :, :], data[:, :, 3, :, :]
        )
        # Energy
        data[:, :, 4, :, :] .= data[:, :, 5, :, :]
    end
    data_view = @view data[:, :, 1:4, :, :]
    return copy(data_view)
end
function get_trajectories(data::AbstractArray{Float32,5})
    density = selectdim(data, 3, 1:1)
    velocity = selectdim(data, 3, 2:3)
    energy = selectdim(data, 3, 5:5)
    momentum = density .* velocity
    trajectories = cat(density, momentum, energy; dims=3)
    return trajectories
end
## Switch from (M, Px, Py, E) to (M, Vx, Vy, P, E)
function get_data(trajectories::AbstractArray{Float32,5})
    density = selectdim(trajectories, 3, 1:1)
    momentum = selectdim(trajectories, 3, 2:3)
    energy = selectdim(trajectories, 3, 5:5)
    velocity = momentum ./ (momentum .+ 1.0f-6)
    pressure = equation_of_state_pressure(density, velocity, energy)
    data = cat(density, velocity, pressure, energy; dims=3)
    return data
end
