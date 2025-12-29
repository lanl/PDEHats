function plot_eig(name_data::Symbol)
    ##
    name_data = :CE
    ##
    eigvals_UNet = get_eigvals(:UNet, name_data)
    eigvals_ViT = get_eigvals(:ViT, name_data)
    ## Plot
    padding_figure = (1, 1, 1, 1)
    size_figure = (400, 250)
    size_title = 18
    size_label = 16
    size_tick_label = 14
    title = "Leading Eigenvalues"
    label_y = "Leading Eigenvalues"
    ##
    cats_ViT = ones(length(eigvals_ViT))
    cats_UNet = 2.0 * ones(length(eigvals_UNet))
    label = ["ViT", "UNet"]
    ticks_x = (range(1.0f0, 2.0f0, 2), label)
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            ylabel=label_y,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            yscale=log10,
            xticks=ticks_x,
            xminorticksvisible=false,
        )
        boxplot!(ax, cats_ViT, eigvals_ViT; show_outliers=false)
        boxplot!(ax, cats_UNet, eigvals_UNet; show_outliers=false)
        return current_figure()
    end
    # wsave(projectdir(path_save * ".pdf"), fig)
    ##
    return fig
end
##
function get_eigvals(name_model::Symbol, name_data::Symbol)
    ##
    dir_load = projectdir("results/Eigen/$(name_data)/$(name_model)")
    paths_load = PDEHats.find_files_by_suffix(dir_load, "eigen.jld2")
    eigvals = map(paths_load) do p
        return load(p)["vals"]
    end
    ##
    return vcat(eigvals...)
end
