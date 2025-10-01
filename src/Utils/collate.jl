##
function shift_pair(X::Vector{<:AbstractArray{Float32,4}})
    inputs = map(X) do x
        T = size(x, 4)
        return selectdim(x, 4, 1:(T - 1))
    end
    targets = map(X) do x
        T = size(x, 4)
        return selectdim(x, 4, 2:T)
    end
    data_input = stack(inputs)
    data_target = stack(targets)
    ##
    return (data_input, data_target)
end
function shift_pair(X::Vector{<:AbstractArray{Float32,5}})
    inputs = map(X) do x
        T = size(x, 4)
        return selectdim(x, 4, 1:(T - 1))
    end
    targets = map(X) do x
        T = size(x, 4)
        return selectdim(x, 4, 2:T)
    end
    data_input = cat(inputs...; dims=5)
    data_target = cat(targets...; dims=5)
    ##
    return (data_input, data_target)
end
