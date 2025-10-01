##
function hist_1x1(
    vals::AbstractVector{Float32},
    lims_axis::NTuple{2,Float32},
    title::AbstractString,
    label_x::AbstractString,
    label_y::AbstractString;
    path_save::String="dir_save_default/hist_2x2",
    filetype_ext::String=".pdf",
    kwargs...,
)
    ##
    fig = _hist_1x1(vals, lims_axis, title, label_x, label_y; kwargs...)
    wsave(projectdir(path_save * filetype_ext), fig)
    ##
    return fig
end
##
function hist_1x1_with_legend_a(
    vals::AbstractVector{Float32},
    lims_axis::NTuple{2,Float32},
    title::AbstractString,
    label_x::AbstractString,
    label_y::AbstractString;
    size_label_legend::Int=11,
    path_save::String="dir_save_default/hist_2x2_with_legend_a",
    filetype_ext::String=".pdf",
    kwargs...,
)
    ##
    fig = _hist_1x1(vals, lims_axis, title, label_x, label_y; kwargs...)
    #
    value_mean = mean(vals)
    value_std = std(vals)
    label_mean = @sprintf("Mean = %.3e", value_mean)
    label_std = @sprintf("Std = %.3e", value_std)
    #
    ax = Makie.Axis(fig[1, 1])
    hidedecorations!(ax)
    hist!(ax, [1000]; label=label_mean, color=:transparent)
    hist!(ax, [1000]; label=label_std, color=:transparent)
    #
    axislegend(ax; labelsize=size_label_legend)
    ##
    wsave(projectdir(path_save * filetype_ext), fig)
    ##
    return fig
end
##
function hist_1x1_with_legend_b(
    vals::AbstractVector{Float32},
    lims_axis::NTuple{2,Float32},
    title::AbstractString,
    label_x::AbstractString,
    label_y::AbstractString;
    size_label_legend::Int=11,
    path_save::String="dir_save_default/hist_2x2_with_legend_b",
    filetype_ext::String=".pdf",
    kwargs...,
)
    ##
    fig = _hist_1x1(vals, lims_axis, title, label_x, label_y; kwargs...)
    #
    N_under = length(findall(v -> v < first(lims_axis), vals))
    N_over = length(findall(v -> v > last(lims_axis), vals))
    percent_out = 100.0f0 * (N_under + N_over) / length(vals)
    #
    value_mean = mean(vals)
    value_std = std(vals)
    value_worst =
        abs(minimum(vals)) > abs(maximum(vals)) ? minimum(vals) : maximum(vals)
    label_mean = @sprintf("Mean = %.3e", value_mean)
    label_std = @sprintf("Std = %.3e", value_std)
    label_worst = @sprintf("Worst = %.3e", value_worst)
    label_out = @sprintf("Excluded = %.2f%%", percent_out)
    #
    ax = Makie.Axis(fig[1, 1])
    hidedecorations!(ax)
    hist!(ax, [1000]; label=label_mean, color=:transparent)
    hist!(ax, [1000]; label=label_std, color=:transparent)
    hist!(ax, [1000]; label=label_worst, color=:transparent)
    hist!(ax, [1000]; label=label_out, color=:transparent)
    #
    axislegend(ax; labelsize=size_label_legend)
    ##
    wsave(projectdir(path_save * filetype_ext), fig)
    ##
    return fig
end
##
function _hist_1x1(
    vals::AbstractVector{Float32},
    lims_axis::NTuple{2,Float32},
    title::AbstractString,
    label_x::AbstractString,
    label_y::AbstractString;
    size_fig::NTuple{2,Int}=(200, 200),
    size_title::Int=10,
    size_label::Int=8,
    N_bins::Int=64,
    y_max::AbstractFloat=1.0f0,
    padding_figure::NTuple{4,Int}=(1, 8, 1, 1),
)
    ##
    vals_min = minimum(vals)
    vals_max = maximum(vals)
    bins_lims = vcat(
        -Inf, range(first(lims_axis), last(lims_axis), N_bins), Inf
    )
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        #
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            xlabel=label_x,
            xlabelsize=size_label,
            ylabel=label_y,
            ylabelsize=size_label,
        )
        xlims!(ax, first(lims_axis), last(lims_axis))
        ylims!(ax, 0, y_max)
        #
        hist!(ax, vals; normalization=:probability, bins=bins_lims)
        #
        return current_figure()
    end
    ##
    return fig
end
