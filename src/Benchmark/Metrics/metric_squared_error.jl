##
function metric_squared_error(
    x::AbstractArray{Float32,N}, y::AbstractArray{Float32,N}
) where {N}
    scores = abs2.(x .- y)
    return scores
end
function metric_squared_error(b::Benchmark)
    scores = metric_squared_error(b.targets, b.preds)
    return scores
end
function plot(
    b::Benchmark,
    f::typeof(metric_squared_error);
    dir_save="dir_save_default/benchmark/",
)
    ##
    name_metric = string(nameof(f))
    dir_save = dir_save * name_metric * "/"
    ##
    scores = summary_5pt(b, f; dir_save=dir_save * "summary_5pt/")
    ## Hist
    vals_hist = vec(scores)
    lims_axis = (minimum(vals_hist), maximum(vals_hist))
    title = "Error Distribution (MSE)"
    label_x = "Error"
    label_y = "Probability Fraction"
    hist_1x1_with_legend_b(
        vals_hist,
        lims_axis,
        title,
        label_x,
        label_y;
        path_save=dir_save * "hist",
    )
    ## Scatter
    (Lx, Ly, F, T, B) = size(scores)
    scores_p = permutedims(scores, (4, 1, 2, 5, 3))
    scores_p_r = reshape(scores_p, (T, Lx * Ly * B, F))
    supertitle = "Prediction Error"
    label_x = "Time"
    label_y = "MSE"
    titles = ("Mass", "Momentum (Horizontal)", "Momentum (Vertical)", "Energy")
    padding_figure = (2, 5, 1, 1)
    ##
    scatterbars_4x1(
        scores_p_r,
        supertitle,
        titles,
        label_x,
        label_y;
        padding_figure=padding_figure,
        path_save=dir_save * "scatterbar",
    )
    ##
    return nothing
end
