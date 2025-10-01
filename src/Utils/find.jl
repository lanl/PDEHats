##
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
##
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
