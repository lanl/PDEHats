function plot_eig()
    ##
    name_datas = (:CE, :NS)
    for name_data in name_datas
        try
            plot_eig(name_data)
        catch e
        end
    end
    ##
    return nothing
end
function plot_eig(name_data::Symbol)
    ##
    eigvals_UNet = get_eigvals(:UNet, name_data)
    eigvals_ViT = get_eigvals(:ViT, name_data)
    ## Plot
    padding_figure = (1, 5, 1, 1)
    size_figure = (400, 250)
    size_title = 20
    size_label = 20
    size_tick_label = 16
    title = "NTK Spectral Comparison ($(name_data))"
    label_y = "Largest Eigenvalues"
    ##
    cats_ViT = ones(length(eigvals_ViT))
    cats_UNet = 2.0 * ones(length(eigvals_UNet))
    label = ["ViT", "UNet"]
    ticks_x = (range(1.0f0, 2.0f0, 2), label)
    lims_y = (10^(6.25), 10^(8.25))
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            ylabel=label_y,
            ylabelsize=size_label,
            xticklabelsize=size_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            yscale=log10,
            xticks=ticks_x,
            xminorticksvisible=false,
        )
        ylims!(ax, lims_y)
        boxplot!(
            ax, cats_ViT, eigvals_ViT; show_outliers=false, show_notch=false
        )
        boxplot!(
            ax,
            cats_UNet,
            eigvals_UNet;
            show_outliers=false,
            show_notch=false,
        )
        return current_figure()
    end
    ##
    path_save = plotsdir("HatMatrix/$(name_data)/eig.pdf")
    wsave(path_save, fig)
    ##
    return fig
end
##
function get_eigvals(name_model::Symbol, name_data::Symbol)
    ##
    dir_load = projectdir("results/Eigen/$(name_data)/$(name_model)")
    paths_load = PDEHats.find_files_by_suffix(dir_load, "eigen_max.jld2")
    eigvals = map(paths_load) do p
        return maximum(load(p)["vals"])
    end
    ##
    return vcat(eigvals...)
end
