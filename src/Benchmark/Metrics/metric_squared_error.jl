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
    dir_save=projectdir("dir_save_default/benchmark/"),
)
    ##
    name_metric = string(nameof(f))
    dir_save = dir_save * name_metric * "/"
    ##
    scores = summary_5pt(b, f; dir_save=dir_save * "summary_5pt/")
    ## Hist
    vals_hist = vec(scores)
    lim_min = quantile(vals_hist, 0.05f0)
    lim_max = quantile(vals_hist, 0.75f0)
    lims_axis = (lim_min, lim_max)
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
    vals_scatter = map(f -> scores_p_r[:, :, f], Tuple(collect(1:F)))
    ##
    scatterbars(vals_scatter; dir_save=dir_save)
    ##
    return nothing
end
function scatterbars(
    vals::NTuple{4,Matrix{Float32}};
    dir_save::String=projectdir("dir_save_default/benchmark/"),
)
    supertitle = "Prediction Error"
    label_x = "Time"
    label_y = "MSE"
    titles = ("Mass", "Momentum (Horizontal)", "Momentum (Vertical)", "Energy")
    padding_figure = (2, 5, 1, 1)
    scatterbars(
        vals,
        supertitle,
        titles,
        label_x,
        label_y;
        padding_figure=padding_figure,
        path_save=dir_save * "scatterbars_4x1",
    )
    return nothing
end
