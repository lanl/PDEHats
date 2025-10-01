##
function scatterbars_4x1(
    vals::AbstractArray{Float32,3},
    supertitle::String,
    titles::NTuple{4,String},
    label_x::String,
    label_y::String;
    size_supertitle::Int=16,
    size_title::Int=12,
    size_label::Int=12,
    size_fig::NTuple{2,Int}=(250, 800),
    width_whisker::Int=10,
    path_save::String="dir_save_default/scatterbars_4x1",
    filetype_ext::String=".pdf",
    padding_figure::NTuple{4,Int}=(1, 2, 1, 1),
)
    ##
    T = size(vals, 1)
    ##
    means = mean(vals; dims=2)
    stds = std(vals; dims=2)
    ##
    val_1 = vec(selectdim(means, 3, 1))
    val_2 = vec(selectdim(means, 3, 2))
    val_3 = vec(selectdim(means, 3, 3))
    val_4 = vec(selectdim(means, 3, 4))
    ##
    std_1 = vec(selectdim(stds, 3, 1))
    std_2 = vec(selectdim(stds, 3, 2))
    std_3 = vec(selectdim(stds, 3, 3))
    std_4 = vec(selectdim(stds, 3, 4))
    ##
    extrema = maximum(abs.(means) .+ stds)
    ##
    y_lim = maximum(extrema) * 1.05f0
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        supertitle = Label(fig[0, 1], supertitle; fontsize=size_supertitle)
        #
        ax_1 = Makie.Axis(
            fig[1, 1];
            title=titles[1],
            titlesize=size_title,
            ylabel=label_y,
            ylabelsize=size_label,
            xticklabelsvisible=false,
        )
        scatter!(ax_1, val_1)
        errorbars!(ax_1, 1:T, val_1, std_1; whiskerwidth=width_whisker)
        #
        ylims!(ax_1, -y_lim, y_lim)
        ax_2 = Makie.Axis(
            fig[2, 1];
            title=titles[2],
            ylabel=label_y,
            ylabelsize=size_label,
            titlesize=size_title,
            xticklabelsvisible=false,
        )
        ylims!(ax_2, -y_lim, y_lim)
        scatter!(ax_2, val_2)
        errorbars!(ax_2, 1:T, val_2, std_2; whiskerwidth=width_whisker)
        #
        ax_3 = Makie.Axis(
            fig[3, 1];
            title=titles[3],
            titlesize=size_title,
            ylabel=label_y,
            ylabelsize=size_label,
            xticklabelsvisible=false,
        )
        ylims!(ax_3, -y_lim, y_lim)
        scatter!(ax_3, val_3)
        errorbars!(ax_3, 1:T, val_3, std_3; whiskerwidth=width_whisker)
        #
        ax_4 = Makie.Axis(
            fig[4, 1];
            title=titles[4],
            titlesize=size_title,
            ylabel=label_y,
            ylabelsize=size_label,
            xlabel=label_x,
            xlabelsize=size_label,
        )
        ylims!(ax_4, -y_lim, y_lim)
        scatter!(ax_4, val_4)
        errorbars!(ax_4, 1:T, val_4, std_4; whiskerwidth=width_whisker)
        ##
        return current_figure()
    end
    ##
    wsave(projectdir(path_save * filetype_ext), fig)
    ##
    return nothing
end
