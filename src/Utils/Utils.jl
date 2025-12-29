##
include("collate.jl")
include("get_data.jl")
include("dataset.jl")
## Rank aware logging
function _println(should_log::Bool, msg::String)
    return should_log && println("[$(now())] ", msg)
end
function _print(should_log::Bool, msg::String)
    return should_log && print("[$(now())] ", msg)
end
## Naming lexicon
function _nameof(s::String)
    replacements = Dict(
        "loss_mse_scaled" => "Mean Squared Error (Scaled)",
        "loss_ssim_scaled" => "SSIM (Scaled)",
    )
    return haskey(replacements, s) ? replacements[s] : s
end
## Padding
function get_padding_circular(K::Int)
    padding = Tuple(mapfoldl(i -> [cld(i, 2), fld(i, 2)], vcat, (K, K) .- 1))
    return padding
end
## Find
function find_files(dir::String, prefix::String, suffix::String)
    found_files = String[]
    for (root, _, files) in walkdir(dir)
        for file in files
            if startswith(file, prefix) && endswith(file, suffix)
                push!(found_files, joinpath(root, file))
            end
        end
    end
    return found_files
end
function find_files_by_suffix(dir::String, suffix::String)
    found_files = String[]
    for (root, _, files) in walkdir(dir)
        for file in files
            if endswith(file, suffix)
                push!(found_files, joinpath(root, file))
            end
        end
    end
    return found_files
end
## Loading from jld2
function get_keys_jld2(path_jld2::String)
    keys_cfg = jldopen(path_jld2, "r") do file
        return keys(file)
    end
    return keys_cfg
end
function load_keys_jld2(
    path_jld2::String, keys_to_load::NTuple{N,String}
) where {N}
    cfg = Dict(k => load(path_jld2, k) for k in keys_to_load)
    return cfg
end
##
function float_to_string(x::Float32)
    s = @sprintf("%.3e", float(x))
    mant, exp = split(s, 'e')
    intpart, fracpart = split(mant, '.')

    fracpart = rstrip(fracpart, '0')
    mant_new = intpart * fracpart

    exp_new = parse(Int, exp) - length(fracpart)

    out = mant_new * "f" * string(exp_new)
    return out
end
##
