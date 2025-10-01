## Global Conservation Laws (t -> t+1)
function metric_cons_global(b::Benchmark)
    scores = get_Q_global_flucts(b.inputs, b.preds)
    return scores
end
## Global Conservation Laws (Rollout)
function metric_cons_global_rollout(b::Benchmark)
    scores = get_Q_global_flucts(b.preds)
    return scores
end
##
function plot(
    b::Benchmark,
    f::Union{typeof(metric_cons_global),typeof(metric_cons_global_rollout)};
    dir_save="dir_save_default/benchmark/",
)
    ##
    name_metric = string(nameof(f))
    dir_save = dir_save * name_metric * "/"
    ##
    scores = summary_5pt(b, f; dir_save=dir_save * "summary_5pt/")
    ## Scatter (Lx, Ly = 1)
    (Lx, Ly, F, T, B) = size(scores)
    scores_p = permutedims(scores, (4, 1, 2, 5, 3))
    scores_p_r = reshape(scores_p, (T, Lx * Ly * B, F))
    supertitle = "Global Conservation Law Errors"
    titles = ("Mass", "Momentum (Horizontal)", "Momentum (Vertical)", "Energy")
    label_x = "Time"
    label_y = "Percent Deviation"
    scatterbars_4x1(
        scores_p_r,
        supertitle,
        titles,
        label_x,
        label_y;
        path_save=dir_save * "scatterbars",
    )
    ## Hist
    vals_r = reshape(scores_p_r, (T * B, F))
    lims_bins = (
        (-2.50f0, 2.50f0),
        (-2.50f0, 2.50f0),
        (-2.50f0, 2.50f0),
        (-2.50f0, 2.50f0),
    )
    label_y = "Probability Fraction"
    label_x_1 = "Percent Deviation"
    label_x_2 = "Typical Relative Deviation"
    hist_2x2_with_legend_b(
        vals_r,
        lims_bins,
        supertitle,
        titles,
        label_x_1,
        label_x_2,
        label_y;
        path_save=dir_save * "hist",
    )
    ##
    return nothing
end
