## Saves Input-Input distances/diffs
function save_diffs()
    ##
    names_data = (:CE, :NS)
    names_model = (:UNet, :ViT)
    diff_fns = (diff_mse, diff_smse)
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
        (; idx_rp=7, idx_crp=7, idx_rpui=7),
        (; idx_rp=8, idx_crp=8, idx_rpui=8),
        (; idx_rp=9, idx_crp=9, idx_rpui=9),
        (; idx_rp=10, idx_crp=10, idx_rpui=10),
        (; idx_rp=11, idx_crp=11, idx_rpui=11),
        (; idx_rp=12, idx_crp=12, idx_rpui=12),
        (; idx_rp=13, idx_crp=13, idx_rpui=13),
        (; idx_rp=14, idx_crp=14, idx_rpui=14),
        (; idx_rp=15, idx_crp=15, idx_rpui=15),
        (; idx_rp=16, idx_crp=16, idx_rpui=16),
        (; idx_rp=17, idx_crp=17, idx_rpui=17),
        (; idx_rp=18, idx_crp=18, idx_rpui=18),
    )
    ##
    for name_data in names_data
        if name_data == :CE
            epoch = 150
        elseif name_data == :NS
            epoch = 100
        end
        for name_model in names_model
            for diff_fn in diff_fns
                for seed in seeds
                    for idx_NT in idx_NTs
                        get_diffs(
                            name_model,
                            name_data,
                            seed,
                            epoch,
                            idx_NT;
                            diff_fn=diff_fn,
                        )
                    end
                end
            end
        end
    end
    return nothing
end
##
function get_diffs(
    name_model::Symbol,
    name_data::Symbol,
    seed::Int,
    epoch::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    diff_fn=diff_mse,
    ratio_train::AbstractFloat=0.65f0,
    ratio_val::AbstractFloat=0.05f0,
    T_max::Int=17,
)
    ## Unpack
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    ## Saving Pathing
    path_save = projectdir(
        "results/Diffs/$(name_data)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)/$(nameof(diff_fn))/",
    )
    name_file_save = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    ## Load Batch
    obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
    (input, target) = obs
    (Lx, Ly, F, T, N) = size(input)
    #
    println("Getting Diffs: $(name_file_save)")
    dict_result, path_dict_result =
        produce_or_load((;), path_save; filename=name_file_save) do _
            diffs =
                map(Iterators.product(1:T, 1:N, 1:T, 1:N)) do (T1, N1, T2, N2)
                    input_1 = input[:, :, :, T1, N1]
                    input_2 = input[:, :, :, T2, N2]
                    diff = diff_fn(input_1, input_2)
                    return diff
                end
            dict = Dict("diffs" => diffs)
            return dict
        end
    diffs = dict_result["diffs"]
    return diffs
end
##
function diff_mse(
    input_1::AbstractArray{Float32,3}, input_2::AbstractArray{Float32,3}
)
    diff = mean(abs2.(input_1 .- input_2))
    return diff
end
##
function diff_smse(
    input_1::AbstractArray{Float32,3}, input_2::AbstractArray{Float32,3}
)
    scale_1 = sqrt.(mean(abs2, input_1; dims=(1, 2)))
    scale_2 = sqrt.(mean(abs2, input_2; dims=(1, 2)))
    scale = 0.5f0 .* (scale_1 .+ scale_2) .+ 1.0f-6
    diff = mean(abs2.(input_1 .- input_2) ./ scale)
    return diff
end
##
