##
function save_nc_data(data::Array{Float32,5}, path_save::String)
    NCDataset(path_save * ".nc", "c") do ds
        data_py = permutedims(data, (2, 1, 3, 4, 5))
        ds.dim["dim1"] = size(data_py, 1)
        ds.dim["dim2"] = size(data_py, 2)
        ds.dim["dim3"] = size(data_py, 3)
        ds.dim["dim4"] = size(data_py, 4)
        ds.dim["dim5"] = size(data_py, 5)
        if occursin("RM", path_save)
            var = defVar(
                ds,
                "solution",
                Float32,
                ("dim1", "dim2", "dim3", "dim4", "dim5"),
            )
        else
            var = defVar(
                ds, "data", Float32, ("dim1", "dim2", "dim3", "dim4", "dim5")
            )
        end
        return var[:] = data_py
    end
    return nothing
end
##
function save_nc(data::Array{Float32,3}, path_save::String)
    NCDataset(path_save * ".nc", "c") do ds
        ds.dim["dim1"] = size(data, 1)
        ds.dim["dim2"] = size(data, 2)
        ds.dim["dim3"] = size(data, 3)
        var = defVar(ds, "data", Float32, ("dim1", "dim2", "dim3"))
        return var[:] = data
    end
    return nothing
end
