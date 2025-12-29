function save_obs_batch()
    names_data = (:CE, :NS)
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    for name_data in names_data
        for seed in seeds
            for idx_NT in idx_NTs
                get_obs_batch(name_data, seed, idx_NT)
            end
        end
    end
    return nothing
end
##
function get_obs_batch(
    name_data::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    ratio_train::AbstractFloat=0.65f0,
    ratio_val::AbstractFloat=0.05f0,
    T_max::Int=17,
)
    ##
    if name_data == :CE
        classes_data = ("CE-RP", "CE-CRP", "CE-RPUI")
    elseif name_data == :NS
        classes_data = ("NS-BB", "NS-Gauss", "NS-Sines")
    else
        throw("only name_data == :CE and :NS are supported")
    end

    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    path_save_0 = "results/HatMatrix/$(name_data)/batches_test"
    path_save_1 = "seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)"
    path_save_2 = "Tmax_$(T_max)"
    path_save = projectdir(join([path_save_0, path_save_1, path_save_2], "/"))

    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    name_file = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    ## Seeding
    rng = Xoshiro(seed)
    ##
    dict, path_dict = produce_or_load((;), path_save; filename=name_file) do _
        ## Dataset
        println("Loading Data")
        ## RP
        trajectory_rp = get_trajectory_test(
            rng,
            classes_data[1],
            idx_rp,
            ratio_train,
            ratio_val;
            T_max=T_max,
        )
        ## CRP
        trajectory_crp = get_trajectory_test(
            rng,
            classes_data[2],
            idx_crp,
            ratio_train,
            ratio_val;
            T_max=T_max,
        )
        ## RPUI
        trajectory_rpui = get_trajectory_test(
            rng,
            classes_data[3],
            idx_rpui,
            ratio_train,
            ratio_val;
            T_max=T_max,
        )
        ## Batch
        trajectories = [trajectory_rp, trajectory_crp, trajectory_rpui]
        obs = PDEHats.shift_pair(trajectories)
        (input, target) = obs
        dict_obs = Dict("input" => input, "target" => target)
        return dict_obs
    end
    ##
    input = dict["input"]
    target = dict["target"]
    obs = (input, target)
    ##
    return obs
end
function get_trajectory_test(
    rng::AbstractRNG,
    name_data::String,
    idx::Int,
    ratio_train::AbstractFloat,
    ratio_val::AbstractFloat;
    T_max::Int=17,
)
    (idx_train, idx_val, idx_test) = PDEHats.get_idx_trajectories_split(
        rng, name_data, ratio_train, ratio_val
    )
    trajectories_test = PDEHats.get_trajectories(
        name_data, idx_test; T_max=T_max
    )
    trajectory_test = copy(selectdim(trajectories_test, 5, idx:idx))
    return trajectory_test
end
