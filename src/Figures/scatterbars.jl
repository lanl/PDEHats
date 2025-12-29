##
function scatterbars(
    vals::NTuple{4,Matrix{Float32}},
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
    padding_figure::NTuple{4,Int}=(1, 2, 1, 1),
)
    ##
    means = map(v -> vec(mean(v; dims=2)), vals)
    stds = map(v -> vec(std(v; dims=2)), vals)
    Ts = map(v -> size(v, 1), vals)
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
        scatter!(ax_1, means[1])
        errorbars!(ax_1, 1:Ts[1], means[1], stds[1]; whiskerwidth=width_whisker)
        #
        ax_2 = Makie.Axis(
            fig[2, 1];
            title=titles[2],
            ylabel=label_y,
            ylabelsize=size_label,
            titlesize=size_title,
            xticklabelsvisible=false,
        )
        scatter!(ax_2, means[2])
        errorbars!(ax_2, 1:Ts[2], means[2], stds[2]; whiskerwidth=width_whisker)
        #
        ax_3 = Makie.Axis(
            fig[3, 1];
            title=titles[3],
            titlesize=size_title,
            ylabel=label_y,
            ylabelsize=size_label,
            xticklabelsvisible=false,
        )
        scatter!(ax_3, means[3])
        errorbars!(ax_3, 1:Ts[3], means[3], stds[3]; whiskerwidth=width_whisker)
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
        scatter!(ax_4, means[4])
        errorbars!(ax_4, 1:Ts[4], means[4], stds[4]; whiskerwidth=width_whisker)
        ##
        return current_figure()
    end
    ##
    wsave(projectdir(path_save * ".pdf"), fig)
    ##
    return nothing
end
function scatterbars(
    vals::NTuple{2,Matrix{Float32}},
    supertitle::String,
    titles::NTuple{2,String},
    label_x::String,
    label_y::String;
    size_supertitle::Int=16,
    size_title::Int=12,
    size_label::Int=12,
    size_fig::NTuple{2,Int}=(250, 400),
    width_whisker::Int=10,
    path_save::String="dir_save_default/scatterbars_2x1",
    padding_figure::NTuple{4,Int}=(1, 2, 1, 1),
)
    ##
    means = map(v -> vec(mean(v; dims=2)), vals)
    stds = map(v -> vec(std(v; dims=2)), vals)
    Ts = map(v -> size(v, 1), vals)
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
        scatter!(ax_1, means[1])
        errorbars!(ax_1, 1:Ts[1], means[1], stds[1]; whiskerwidth=width_whisker)
        #
        ax_2 = Makie.Axis(
            fig[2, 1];
            title=titles[2],
            ylabel=label_y,
            ylabelsize=size_label,
            titlesize=size_title,
            xticklabelsvisible=false,
        )
        scatter!(ax_2, means[2])
        errorbars!(ax_2, 1:Ts[2], means[2], stds[2]; whiskerwidth=width_whisker)
        ##
        return current_figure()
    end
    ##
    wsave(projectdir(path_save * ".pdf"), fig)
    ##
    return nothing
end
