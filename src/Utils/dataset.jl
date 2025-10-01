## Data Container
struct Dataset
    trajectories_rp_r::AbstractArray{Float32,4}
    trajectories_crp_r::AbstractArray{Float32,4}
    trajectories_rpui_r::AbstractArray{Float32,4}
    size_rp::NTuple{5,Int}
    size_crp::NTuple{5,Int}
    size_rpui::NTuple{5,Int}
    TB_rp::Int
    TB_crp::Int
    TB_rpui::Int
end
function Dataset(
    trajectories_rp::AbstractArray{Float32,5},
    trajectories_crp::AbstractArray{Float32,5},
    trajectories_rpui::AbstractArray{Float32,5},
)
    ## RP
    (Lx, Ly, F, T_max, B) = size(trajectories_rp)
    TB_rp = (T_max - 1) * B
    trajectories_rp_r = reshape(trajectories_rp, (Lx, Ly, F, T_max * B))
    ## CRP
    (Lx, Ly, F, T_max, B) = size(trajectories_crp)
    TB_crp = (T_max - 1) * B
    trajectories_crp_r = reshape(trajectories_crp, (Lx, Ly, F, T_max * B))
    ## RPUI
    (Lx, Ly, F, T_max, B) = size(trajectories_rpui)
    TB_rpui = (T_max - 1) * B
    trajectories_rpui_r = reshape(trajectories_rpui, (Lx, Ly, F, T_max * B))
    ##
    return Dataset(
        trajectories_rp_r,
        trajectories_crp_r,
        trajectories_rpui_r,
        size(trajectories_rp),
        size(trajectories_crp),
        size(trajectories_rpui),
        TB_rp,
        TB_crp,
        TB_rpui,
    )
end
Base.length(d::Dataset) = d.TB_rp + d.TB_crp + d.TB_rpui
function Base.getindex(d::Dataset, idx::Vector{<:Int})
    ## Indexing
    idx_rp = idx[idx .<= d.TB_rp]
    idx_crp = idx[(idx .> d.TB_rp) .&& (idx .<= d.TB_rp + d.TB_crp)] .- d.TB_rp
    idx_rpui = idx[idx .> d.TB_rp + d.TB_crp] .- (d.TB_rp + d.TB_crp)
    ## Input
    trajectories_rp_input = selectdim(d.trajectories_rp_r, 4, idx_rp)
    trajectories_crp_input = selectdim(d.trajectories_crp_r, 4, idx_crp)
    trajectories_rpui_input = selectdim(d.trajectories_rpui_r, 4, idx_rpui)
    ## Target
    trajectories_rp_target = selectdim(d.trajectories_rp_r, 4, idx_rp .+ 1)
    trajectories_crp_target = selectdim(d.trajectories_crp_r, 4, idx_crp .+ 1)
    trajectories_rpui_target = selectdim(
        d.trajectories_rpui_r, 4, idx_rpui .+ 1
    )
    ## Concat
    trajectories_input = cat(
        trajectories_rp_input,
        trajectories_crp_input,
        trajectories_rpui_input;
        dims=4,
    )
    trajectories_target = cat(
        trajectories_rp_target,
        trajectories_crp_target,
        trajectories_rpui_target;
        dims=4,
    )
    ##
    return trajectories_input, trajectories_target
end
##
function get_datasets(
    rng::AbstractRNG,
    ratio_train::AbstractFloat,
    ratio_val::AbstractFloat;
    T_max::Int=21,
    p::AbstractFloat=1.0f0,
)
    ## RP
    trajectories_rp = get_trajectories(rng, "RP"; T_max=T_max, p=p)
    (trajectories_rp_train, trajectories_rp_val, trajectories_rp_test) = splitobs(
        rng, trajectories_rp; at=(ratio_train, ratio_val), shuffle=true
    )
    ## CRP
    trajectories_crp = get_trajectories(rng, "CRP"; T_max=T_max, p=p)
    (trajectories_crp_train, trajectories_crp_val, trajectories_crp_test) = splitobs(
        rng, trajectories_crp; at=(ratio_train, ratio_val), shuffle=true
    )
    ## RPUI
    trajectories_rpui = get_trajectories(rng, "RPUI"; T_max=T_max, p=p)
    (trajectories_rpui_train, trajectories_rpui_val, trajectories_rpui_test) = splitobs(
        rng, trajectories_rpui; at=(ratio_train, ratio_val), shuffle=true
    )
    ##
    dataset_train = Dataset(
        trajectories_rp_train, trajectories_crp_train, trajectories_rpui_train
    )
    dataset_val = Dataset(
        trajectories_rp_val, trajectories_crp_val, trajectories_rpui_val
    )
    dataset_test = Dataset(
        trajectories_rp_test, trajectories_crp_test, trajectories_rpui_test
    )
    ##
    return dataset_train, dataset_val, dataset_test
end
