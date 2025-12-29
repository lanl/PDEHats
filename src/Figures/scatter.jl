function scatter(
    vals::AbstractVector{Float32},
    title::String,
    label_x::String,
    label_y::String;
    padding_figure::NTuple{4,Int}=(1, 1, 1, 1),
    dir_save::String=projectdir("dir_save_default/"),
    name_save::String="scatter_1x1",
)
    ## Log
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure()
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            yscale=log10,
        )
        scatter!(ax, vals)
        return current_figure()
    end
    wsave(projectdir(dir_save * "$(name_save)_log.pdf"), fig)
    ## Linear
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure()
        ax = Makie.Axis(fig[1, 1]; title=title, xlabel=label_x, ylabel=label_y)
        scatter!(ax, vals)
        return current_figure()
    end
    wsave(projectdir(dir_save * "$(name_save)_linear.pdf"), fig)
    ##
    return nothing
end
