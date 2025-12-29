##
function get_data(name_data::Symbol; T_max::Int=21)
    if name_data == :CE_TEST
        ## Testing Slice [32, 32, F, 7, 4]
        ## F = 5
        data = load(datadir("sim_test/CE-TEST.jld2"))["data_test"]
    elseif name_data == :NS_TEST
        ## Testing Slice [32, 32, F, 7, 4], F = 2
        ## F = 2
        data = load(datadir("sim_test/NS-TEST.jld2"))["data_test"]
    else
        ## Full Dataset [128, 128, F, T_max, N]
        # CE: F = 5
        # NS: F = 2
        data = _get_data(name_data; T_max=T_max)
    end
    ##
    return data
end
function _get_data(name_data::Symbol; T_max::Int=21)
    filepath = datadir("sim_pro/$(name_data).nc")
    if occursin("CE", name_data)
        data = NCDataset(filepath, "r") do ds
            return ds["data"][:, :, :, 1:T_max, :]
        end # ds is closed
    elseif occursin("NS", name_data)
        data = NCDataset(filepath, "r") do ds
            return ds["velocity"][:, :, 1:2, 1:T_max, :]
        end # ds is closed
    end
    return data
end
## Switch from (M, Vx, Vy, P, E) to (M, Px, Py, E)
function get_trajectories(name_data::Symbol; T_max::Int=21)
    data = get_data(name_data; T_max=T_max)
    names_CE = (:CE_RP, :CE_CRP, :CE_RPUI, :CE_TEST)
    if (name_data in names_CE)
        return get_trajectories(data)
    else
        return data
    end
end
function get_trajectories(data::AbstractArray{Float32,5})
    @assert size(data, 3) == 5
    density = selectdim(data, 3, 1:1)
    velocity = selectdim(data, 3, 2:3)
    energy = selectdim(data, 3, 5:5)
    momentum = density .* velocity
    trajectories = cat(density, momentum, energy; dims=3)
    return trajectories
end
## Memory Management
function get_trajectories(name_data::String, idx::Vector{Int}; T_max::Int=21)
    filepath = datadir("sim_pro/$(name_data).nc")
    if occursin("CE", name_data)
        data = NCDataset(filepath, "r") do ds
            return ds["data"][:, :, :, 1:T_max, idx]
        end # ds is closed
    elseif occursin("NS", name_data)
        data = NCDataset(filepath, "r") do ds
            return ds["velocity"][:, :, 1:2, 1:T_max, idx]
        end # ds is closed
    end
    trajectories = copy(get_trajectories!(data))
    GC.gc()
    return trajectories
end
function get_trajectories!(data::AbstractArray{Float32,5})
    if size(data, 3) == 5
        @views data[:, :, 2:3, :, :] .=
            data[:, :, 1:1, :, :] .* data[:, :, 2:3, :, :]
        trajectories = selectdim(data, 3, [1, 2, 3, 5])
    elseif size(data, 3) == 2
        trajectories = data
    else
        throw("unexpected data shape")
    end
    return trajectories
end
