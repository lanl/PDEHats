##
function hist_2x2(
    vals::AbstractArray{Float32,2},
    lims_axis::NTuple{4,NTuple{2,Float32}},
    supertitle::String,
    titles::NTuple{4,String},
    label_x_1::String,
    label_x_2::String,
    label_y::String;
    path_save::String="dir_save_default/hist_2x2",
    filetype_ext::String=".pdf",
    save_dict::Bool=false,
    kwargs...,
)
    ##
    fig = _hist_2x2(
        vals,
        lims_axis,
        supertitle,
        titles,
        label_x_1,
        label_x_2,
        label_y;
        kwargs...,
    )
    ##
    if save_dict
        dict = Dict("vals" => vals)
        tagsave(projectdir(path_save * ".jld2"), dict)
    end
    ##
    wsave(projectdir(path_save * filetype_ext), fig)
    ##
    return nothing
end
##
function hist_2x2_with_legend_a(
    vals::AbstractArray{Float32,2},
    lims_axis::NTuple{4,NTuple{2,Float32}},
    supertitle::String,
    titles::NTuple{4,String},
    label_x_1::String,
    label_x_2::String,
    label_y::String;
    path_save::String="dir_save_default/hist_2x2",
    filetype_ext::String=".pdf",
    size_label_legend::Int=11,
    save_dict::Bool=false,
    kwargs...,
)
    ##
    fig = _hist_2x2(
        vals,
        lims_axis,
        supertitle,
        titles,
        label_x_1,
        label_x_2,
        label_y;
        kwargs...,
    )
    # Unpack vals
    vals_11 = vec(selectdim(vals, 2, 1))
    vals_12 = vec(selectdim(vals, 2, 2))
    vals_21 = vec(selectdim(vals, 2, 3))
    vals_22 = vec(selectdim(vals, 2, 4))
    #
    values_11_mean = mean(vals_11)
    values_12_mean = mean(vals_12)
    values_21_mean = mean(vals_21)
    values_22_mean = mean(vals_22)
    #
    values_11_std = std(vals_11)
    values_12_std = std(vals_12)
    values_21_std = std(vals_21)
    values_22_std = std(vals_22)
    #
    ax_11 = Makie.Axis(fig[1, 1])
    hidedecorations!(ax_11)
    ax_12 = Makie.Axis(fig[1, 2])
    hidedecorations!(ax_12)
    ax_21 = Makie.Axis(fig[2, 1])
    hidedecorations!(ax_21)
    ax_22 = Makie.Axis(fig[2, 2])
    hidedecorations!(ax_22)

    label_mean_11 = @sprintf("Mean = %.3e", values_11_mean)
    label_std_11 = @sprintf("Std = %.3e", values_11_std)
    hist!(ax_11, [1000]; label=label_mean_11, color=:transparent)
    hist!(ax_11, [1000]; label=label_std_11, color=:transparent)
    axislegend(ax_11; labelsize=size_label_legend)

    label_mean_12 = @sprintf("Mean = %.3e", values_12_mean)
    label_std_12 = @sprintf("Std = %.3e", values_12_std)
    hist!(ax_12, [1000]; label=label_mean_12, color=:transparent)
    hist!(ax_12, [1000]; label=label_std_12, color=:transparent)
    axislegend(ax_12; labelsize=size_label_legend)

    label_mean_21 = @sprintf("Mean = %.3e", values_21_mean)
    label_std_21 = @sprintf("Std = %.3e", values_21_std)
    hist!(ax_21, [1000]; label=label_mean_21, color=:transparent)
    hist!(ax_21, [1000]; label=label_std_21, color=:transparent)
    axislegend(ax_21; labelsize=size_label_legend)

    label_mean_22 = @sprintf("Mean = %.3e", values_22_mean)
    label_std_22 = @sprintf("Std = %.3e", values_22_std)
    hist!(ax_22, [1000]; label=label_mean_22, color=:transparent)
    hist!(ax_22, [1000]; label=label_std_22, color=:transparent)
    axislegend(ax_22; labelsize=size_label_legend)
    ##
    if save_dict
        dict = Dict("vals" => vals)
        tagsave(projectdir(path_save * ".jld2"), dict)
    end
    ##
    wsave(projectdir(path_save * filetype_ext), fig)
    ##
    return fig
end
function hist_2x2_with_legend_b(
    vals::AbstractArray{Float32,2},
    lims_axis::NTuple{4,NTuple{2,Float32}},
    supertitle::String,
    titles::NTuple{4,String},
    label_x_1::String,
    label_x_2::String,
    label_y::String;
    path_save::String="dir_save_default/hist_2x2",
    filetype_ext::String=".pdf",
    size_label_legend::Int=11,
    save_dict::Bool=false,
    kwargs...,
)
    ##
    fig = _hist_2x2(
        vals,
        lims_axis,
        supertitle,
        titles,
        label_x_1,
        label_x_2,
        label_y;
        kwargs...,
    )
    # Unpack vals
    vals_11 = vec(selectdim(vals, 2, 1))
    vals_12 = vec(selectdim(vals, 2, 2))
    vals_21 = vec(selectdim(vals, 2, 3))
    vals_22 = vec(selectdim(vals, 2, 4))
    #
    values_11_mean = mean(vals_11)
    values_12_mean = mean(vals_12)
    values_21_mean = mean(vals_21)
    values_22_mean = mean(vals_22)
    #
    values_11_std = std(vals_11)
    values_12_std = std(vals_12)
    values_21_std = std(vals_21)
    values_22_std = std(vals_22)
    #
    values_11_worst = if abs(minimum(vals_11)) > maximum(vals_11)
        minimum(vals_11)
    else
        maximum(vals_11)
    end
    values_12_worst = if abs(minimum(vals_12)) > maximum(vals_12)
        minimum(vals_12)
    else
        maximum(vals_12)
    end
    values_21_worst = if abs(minimum(vals_21)) > maximum(vals_21)
        minimum(vals_21)
    else
        maximum(vals_21)
    end
    values_22_worst = if abs(minimum(vals_22)) > maximum(vals_22)
        minimum(vals_22)
    else
        maximum(vals_22)
    end
    ## Unpack axis limits
    lims_axis_11 = lims_axis[1]
    lims_axis_12 = lims_axis[2]
    lims_axis_21 = lims_axis[3]
    lims_axis_22 = lims_axis[4]
    # Compute excluded
    N_under_11 = length(findall(v -> v < first(lims_axis_11), vals_11))
    N_over_11 = length(findall(v -> v > last(lims_axis_11), vals_11))
    percent_out_11 = 100.0f0 * (N_under_11 + N_over_11) / length(vals_11)

    N_under_12 = length(findall(v -> v < first(lims_axis_12), vals_12))
    N_over_12 = length(findall(v -> v > last(lims_axis_12), vals_12))
    percent_out_12 = 100.0f0 * (N_under_12 + N_over_12) / length(vals_12)

    N_under_21 = length(findall(v -> v < first(lims_axis_21), vals_21))
    N_over_21 = length(findall(v -> v > last(lims_axis_21), vals_21))
    percent_out_21 = 100.0f0 * (N_under_21 + N_over_21) / length(vals_21)

    N_under_22 = length(findall(v -> v < first(lims_axis_22), vals_22))
    N_over_22 = length(findall(v -> v > last(lims_axis_22), vals_22))
    percent_out_22 = 100.0f0 * (N_under_22 + N_over_22) / length(vals_22)
    #
    ax_11 = Makie.Axis(fig[1, 1])
    hidedecorations!(ax_11)
    ax_12 = Makie.Axis(fig[1, 2])
    hidedecorations!(ax_12)
    ax_21 = Makie.Axis(fig[2, 1])
    hidedecorations!(ax_21)
    ax_22 = Makie.Axis(fig[2, 2])
    hidedecorations!(ax_22)

    label_worst_11 = @sprintf("Worst = %.3e", values_11_worst)
    label_mean_11 = @sprintf("Mean = %.3e", values_11_mean)
    label_std_11 = @sprintf("Std = %.3e", values_11_std)
    label_out_11 = @sprintf("Excluded = %.2f%%", percent_out_11)
    hist!(ax_11, [1000]; label=label_mean_11, color=:transparent)
    hist!(ax_11, [1000]; label=label_std_11, color=:transparent)
    hist!(ax_11, [1000]; label=label_worst_11, color=:transparent)
    hist!(ax_11, [1000]; label=label_out_11, color=:transparent)
    axislegend(ax_11; labelsize=size_label_legend)

    label_worst_12 = @sprintf("Worst = %.3e", values_12_worst)
    label_mean_12 = @sprintf("Mean = %.3e", values_12_mean)
    label_std_12 = @sprintf("Std = %.3e", values_12_std)
    label_out_12 = @sprintf("Excluded = %.2f%%", percent_out_12)
    hist!(ax_12, [1000]; label=label_mean_12, color=:transparent)
    hist!(ax_12, [1000]; label=label_std_12, color=:transparent)
    hist!(ax_12, [1000]; label=label_worst_12, color=:transparent)
    hist!(ax_12, [1000]; label=label_out_12, color=:transparent)
    axislegend(ax_12; labelsize=size_label_legend)

    label_worst_21 = @sprintf("Worst = %.3e", values_21_worst)
    label_mean_21 = @sprintf("Mean = %.3e", values_21_mean)
    label_std_21 = @sprintf("Std = %.3e", values_21_std)
    label_out_21 = @sprintf("Excluded = %.2f%%", percent_out_21)
    hist!(ax_21, [1000]; label=label_mean_21, color=:transparent)
    hist!(ax_21, [1000]; label=label_std_21, color=:transparent)
    hist!(ax_21, [1000]; label=label_worst_21, color=:transparent)
    hist!(ax_21, [1000]; label=label_out_21, color=:transparent)
    axislegend(ax_21; labelsize=size_label_legend)

    label_worst_22 = @sprintf("Worst = %.3e", values_22_worst)
    label_mean_22 = @sprintf("Mean = %.3e", values_22_mean)
    label_std_22 = @sprintf("Std = %.3e", values_22_std)
    label_out_22 = @sprintf("Excluded = %.2f%%", percent_out_22)
    hist!(ax_22, [1000]; label=label_mean_22, color=:transparent)
    hist!(ax_22, [1000]; label=label_std_22, color=:transparent)
    hist!(ax_22, [1000]; label=label_worst_22, color=:transparent)
    hist!(ax_22, [1000]; label=label_out_22, color=:transparent)
    axislegend(ax_22; labelsize=size_label_legend)
    ##
    if save_dict
        dict = Dict("vals" => vals)
        tagsave(projectdir(path_save * ".jld2"), dict)
    end
    ##
    wsave(projectdir(path_save * filetype_ext), fig)
    ##
    return fig
end
function _hist_2x2(
    vals::AbstractArray{Float32,2},
    lims_axis::NTuple{4,NTuple{2,Float32}},
    supertitle::String,
    titles::NTuple{4,String},
    label_x_1::String,
    label_x_2::String,
    label_y::String;
    size_supertitle::Int=20,
    size_title::Int=12,
    size_label_x::Int=12,
    size_label_y::Int=12,
    size_fig::NTuple{2,Int}=(500, 500),
    N_bins::Int=64,
    padding_figure::NTuple{4,Int}=(1, 15, 1, 1),
    y_max::AbstractFloat=1.0f0,
)
    ## Unpack axis limits
    lims_axis_11 = lims_axis[1]
    lims_axis_12 = lims_axis[2]
    lims_axis_21 = lims_axis[3]
    lims_axis_22 = lims_axis[4]
    ## Unpack vals
    vals_11 = vec(selectdim(vals, 2, 1))
    vals_12 = vec(selectdim(vals, 2, 2))
    vals_21 = vec(selectdim(vals, 2, 3))
    vals_22 = vec(selectdim(vals, 2, 4))
    ## Make axis ranges
    range_bins_11 = range(first(lims_axis_11), last(lims_axis_11), N_bins)
    range_bins_12 = range(first(lims_axis_12), last(lims_axis_12), N_bins)
    range_bins_21 = range(first(lims_axis_21), last(lims_axis_21), N_bins)
    range_bins_22 = range(first(lims_axis_22), last(lims_axis_22), N_bins)
    # Add bins to infinity
    range_bins_11 = vcat(-Inf, collect(range_bins_11), Inf)
    range_bins_12 = vcat(-Inf, collect(range_bins_12), Inf)
    range_bins_21 = vcat(-Inf, collect(range_bins_21), Inf)
    range_bins_22 = vcat(-Inf, collect(range_bins_22), Inf)
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        #
        Label(fig[0, 1:2], supertitle; fontsize=size_supertitle)
        #
        ax_11 = Makie.Axis(
            fig[1, 1];
            title=titles[1],
            titlesize=size_title,
            ylabel=label_y,
            ylabelsize=size_label_y,
            xlabel=label_x_1,
            xlabelsize=size_label_x,
        )
        ylims!(ax_11, 0, y_max)
        xlims!(ax_11, first(lims_axis_11), last(lims_axis_11))
        hist!(ax_11, vals_11; normalization=:probability, bins=range_bins_11)
        #
        ax_12 = Makie.Axis(
            fig[1, 2];
            title=titles[2],
            titlesize=size_title,
            yticklabelsvisible=false,
            xlabel=label_x_1,
            xlabelsize=size_label_x,
        )
        ylims!(ax_12, 0, y_max)
        xlims!(ax_12, first(lims_axis_12), last(lims_axis_12))
        hist!(ax_12, vals_12; normalization=:probability, bins=range_bins_12)
        colgap!(fig.layout, 1, Relative(0.075))
        #
        ax_21 = Makie.Axis(
            fig[2, 1];
            title=titles[3],
            titlesize=size_title,
            xlabel=label_x_2,
            xlabelsize=size_label_x,
            ylabel=label_y,
            ylabelsize=size_label_y,
        )
        ylims!(ax_21, 0, y_max)
        xlims!(ax_21, first(lims_axis_21), last(lims_axis_21))
        hist!(ax_21, vals_21; normalization=:probability, bins=range_bins_21)
        #
        ax_22 = Makie.Axis(
            fig[2, 2];
            title=titles[4],
            titlesize=size_title,
            xlabel=label_x_2,
            xlabelsize=size_label_x,
            yticklabelsvisible=false,
        )
        ylims!(ax_22, 0, y_max)
        xlims!(ax_22, first(lims_axis_22), last(lims_axis_22))
        hist!(ax_22, vals_22; normalization=:probability, bins=range_bins_22)
        return current_figure()
    end
    ##
    return fig
end
