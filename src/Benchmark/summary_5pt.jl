##
function summary_5pt(
    b::Benchmark,
    f::Function;
    dir_save=projectdir("dir_save_default/summary_5pt/"),
    save_dict::Bool=false,
)
    ##
    scores = f(b)
    losses = map(axes(scores, ndims(scores))) do i
        s = selectdim(scores, ndims(scores), i:i)
        return mean(s)
    end
    ##
    idxs_names_save = get_idxs_5pt_summary(losses)
    map(idxs_names_save) do (idx, _name)
        _score = selectdim(scores, ndims(scores), idx)
        _input = selectdim(b.inputs, ndims(b.inputs), idx)
        _pred = selectdim(b.preds, ndims(b.preds), idx)
        _target = selectdim(b.targets, ndims(b.targets), idx)
        _loss = losses[idx]
        path_save = dir_save * "trajectory_rank=" * _name
        if save_dict
            dict = Dict(
                "score" => _score,
                "input" => _input,
                "target" => _target,
                "pred" => _pred,
                "loss" => _loss,
                "name" => _name,
            )
            tagsave(projectdir(path_save * ".jld2"), dict)
        end
        initial_frame = selectdim(_input, 4, 1:1)
        init_pred = cat(initial_frame, _pred; dims=4)
        init_target = cat(initial_frame, _target; dims=4)
        init_delta = init_pred .- init_target
        supertitle = "Field Evolution"
        titles = ("Prediction", "Target", "Difference")
        labels_y = ("Mass", "Momentum-x", "Momentum-y", "Energy")
        return animate_heatmaps_4x3(
            init_pred,
            init_target,
            init_delta,
            supertitle,
            titles,
            labels_y;
            path_save=path_save,
        )
    end
    ##
    return scores
end
##
function get_idxs_5pt_summary(losses::Vector{Float32})
    N_examples = length(losses)
    idxs_sorted = sortperm(losses)
    idx_best = first(idxs_sorted)
    idx_25th_quartile = idxs_sorted[round(
        Int, median(idxs_sorted) - div(N_examples, 4)
    )]
    idx_median = idxs_sorted[div(N_examples, 2, RoundUp)]
    idx_75th_quartile = idxs_sorted[round(
        Int, median(idxs_sorted) + div(N_examples, 4)
    )]
    idx_worst = last(idxs_sorted)
    names_save = ["1", "25", "50", "75", "100"]
    idxs_save = [
        idx_best, idx_25th_quartile, idx_median, idx_75th_quartile, idx_worst
    ]
    return zip(idxs_save, names_save)
end
##
