## Data Container
struct Dataset
    trajectories_rp::AbstractArray{Float32,5}
    trajectories_crp::AbstractArray{Float32,5}
    trajectories_rpui::AbstractArray{Float32,5}
    length_rp::Int
    length_crp::Int
    length_rpui::Int
    Lx::Int
    Ly::Int
    F::Int
    T_max::Int
end
function Dataset(
    trajectories_rp::AbstractArray{Float32,5},
    trajectories_crp::AbstractArray{Float32,5},
    trajectories_rpui::AbstractArray{Float32,5},
)
    ##
    @assert size(trajectories_rp)[1:4] == size(trajectories_crp)[1:4]
    @assert size(trajectories_crp)[1:4] == size(trajectories_rpui)[1:4]
    (Lx, Ly, F, T_max) = size(trajectories_rp)[1:4]
    ## RP
    B_rp = size(trajectories_rp, 5)
    length_rp = (T_max - 1) * B_rp
    ## CRP
    B_crp = size(trajectories_crp, 5)
    length_crp = (T_max - 1) * B_crp
    ## RPUI
    B_rpui = size(trajectories_rpui, 5)
    length_rpui = (T_max - 1) * B_rpui
    ##
    return Dataset(
        trajectories_rp,
        trajectories_crp,
        trajectories_rpui,
        length_rp,
        length_crp,
        length_rpui,
        Lx,
        Ly,
        F,
        T_max,
    )
end
Base.length(d::Dataset) = d.length_rp + d.length_crp + d.length_rpui
function Base.getindex(d::Dataset, idx::Vector{<:Int})
    ##
    Lx = d.Lx
    Ly = d.Ly
    F = d.F
    T_max = d.T_max
    ## Indexing
    idx_rp = idx[idx .<= d.length_rp]
    idx_crp =
        idx[(idx .> d.length_rp) .&& (idx .<= d.length_rp + d.length_crp)] .-
        d.length_rp
    idx_rpui =
        idx[idx .> d.length_rp + d.length_crp] .- (d.length_rp + d.length_crp)
    ## Input
    inputs_rp_r = selectdim(d.trajectories_rp, 4, 1:(T_max - 1))
    inputs_crp_r = selectdim(d.trajectories_crp, 4, 1:(T_max - 1))
    inputs_rpui_r = selectdim(d.trajectories_rpui, 4, 1:(T_max - 1))
    ## Target
    targets_rp_r = selectdim(d.trajectories_rp, 4, 2:T_max)
    targets_crp_r = selectdim(d.trajectories_crp, 4, 2:T_max)
    targets_rpui_r = selectdim(d.trajectories_rpui, 4, 2:T_max)
    ## Shaping
    inputs_rp = reshape(inputs_rp_r, (Lx, Ly, F, :))
    inputs_crp = reshape(inputs_crp_r, (Lx, Ly, F, :))
    inputs_rpui = reshape(inputs_rpui_r, (Lx, Ly, F, :))
    targets_rp = reshape(targets_rp_r, (Lx, Ly, F, :))
    targets_crp = reshape(targets_crp_r, (Lx, Ly, F, :))
    targets_rpui = reshape(targets_rpui_r, (Lx, Ly, F, :))
    ## Input Selection
    inputs_rp_idx = selectdim(inputs_rp, 4, idx_rp)
    inputs_crp_idx = selectdim(inputs_crp, 4, idx_crp)
    inputs_rpui_idx = selectdim(inputs_rpui, 4, idx_rpui)
    ## Target Selection
    targets_rp_idx = selectdim(targets_rp, 4, idx_rp)
    targets_crp_idx = selectdim(targets_crp, 4, idx_crp)
    targets_rpui_idx = selectdim(targets_rpui, 4, idx_rpui)
    ## Concat
    inputs = cat(inputs_rp_idx, inputs_crp_idx, inputs_rpui_idx; dims=4)
    targets = cat(targets_rp_idx, targets_crp_idx, targets_rpui_idx; dims=4)
    ##
    return inputs, targets
end
##
function get_datasets(
    rng::AbstractRNG,
    name_data::Symbol,
    ratio_train::AbstractFloat,
    ratio_val::AbstractFloat;
    T_max::Int=17,
    should_log::Bool=true,
)
    ##
    if name_data == :CE
        classes_data = ("CE-RP", "CE-CRP", "CE-RPUI")
    elseif name_data == :NS
        classes_data = ("NS-BB", "NS-Gauss", "NS-Sines")
    else
        throw("only name_data == :CE and :NS are supported")
    end
    ## xxx Split rng into three and then replicate before get_trajectories_split
    # seeds = rand(UInt, rng, 3)
    # rngs = map(s -> Xoshiro(s))
    # Cannot do change without breaking reproducibility
    _println(should_log, "Free memory:  $(Sys.free_memory() / 1e9) GB")
    ## RP
    trajectories_rp_train, trajectories_rp_val = get_trajectories(
        rng,
        classes_data[1],
        ratio_train,
        ratio_val;
        T_max=T_max,
        should_log=should_log,
    )
    ## CRP
    trajectories_crp_train, trajectories_crp_val = get_trajectories(
        rng,
        classes_data[2],
        ratio_train,
        ratio_val;
        T_max=T_max,
        should_log=should_log,
    )
    ## RPUI
    trajectories_rpui_train, trajectories_rpui_val = get_trajectories(
        rng,
        classes_data[3],
        ratio_train,
        ratio_val;
        T_max=T_max,
        should_log=should_log,
    )
    ##
    _println(should_log, "Constructing Datasets")
    dataset_train = Dataset(
        trajectories_rp_train, trajectories_crp_train, trajectories_rpui_train
    )
    dataset_val = Dataset(
        trajectories_rp_val, trajectories_crp_val, trajectories_rpui_val
    )
    ##
    return dataset_train, dataset_val
end
## Memory management
function get_idx_trajectories_split(
    rng::AbstractRNG,
    name_data::String,
    ratio_train::AbstractFloat,
    ratio_val::AbstractFloat;
)
    ##
    if occursin("CE", name_data)
        idx_dataset = 1:10_000
    elseif occursin("NS", name_data)
        idx_dataset = 1:20_000
    else
        throw("only name_data == :CE and :NS are supported")
    end
    (idx_train, idx_val, idx_test) =
        copy.(
            splitobs(
                rng, idx_dataset; at=(ratio_train, ratio_val), shuffle=true
            )
        )
    ##
    return idx_train, idx_val, idx_test
end
function get_trajectories(
    rng::AbstractRNG,
    name_data::String,
    ratio_train::AbstractFloat,
    ratio_val::AbstractFloat;
    T_max::Int=17,
    should_log::Bool=true,
)
    _println(should_log, "Loading $(name_data)")
    (idx_train, idx_val, idx_test) = get_idx_trajectories_split(
        rng, name_data, ratio_train, ratio_val
    )
    trajectories_train = SharedArray(
        get_trajectories(name_data, idx_train; T_max=T_max)
    )
    GC.gc()
    trajectories_val = SharedArray(
        get_trajectories(name_data, idx_val; T_max=T_max)
    )
    GC.gc()
    _println(should_log, "Free memory:  $(Sys.free_memory() / 1e9) GB")
    return trajectories_train, trajectories_val
end
