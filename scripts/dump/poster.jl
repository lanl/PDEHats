##
using DrWatson
@quickactivate :PDEHats
##
using CairoMakie, MakiePublication
using Statistics
##
data_RP = PDEHats._get_data("RP")[:, :, 1, end, 1];
data_RPUI = PDEHats._get_data("RPUI")[:, :, 1, end, 20];
data_CRP = PDEHats._get_data("CRP")[:, :, 1, end, 1];
densities = (data_RP, data_RPUI, data_CRP)
titles = ("CE-RP", "CE-RPUI", "CE-CRP")
##
size_title = 20
padding_figure = (5, 5, 5, 5)
size_fig = (400, 400)
for i in 1:3
    density = densities[i]
    title = titles[i]
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            aspect=DataAspect(),
            yticklabelsvisible=false,
            xticklabelsvisible=false,
        )
        vals = density .- mean(density)
        vals = density
        hm = heatmap!(ax, vals; colormap=:berlin100)
        return current_figure()
    end
    wsave(projectdir("density_$(title)_end.pdf"), fig)
end
##
