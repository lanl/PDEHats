function plot_eig()
    ##
    name_datas = (:CE, :NS)
    name_models = (:ViT, :UNet)
    for name_data in name_datas
        for name_model in name_models
            try
                plot_eig_Q(name_model, name_data)
                plot_eig_ratio_Q(name_model, name_data)
            catch e
                println(e)
            end
        end
    end
    ##
    return nothing
end
function plot_eig_Q(name_model::Symbol, name_data::Symbol)
    ##
    if name_data == :CE
        if name_model == :ViT
            epochs = [1, 25, 50, 75, 100, 105, 120, 135, 150]
        elseif name_model == :UNet
            epochs = [1, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
        end
    elseif name_data == :NS
        epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    end
    ##
    eigvals_array = map(
        epoch -> get_eigvals_Q(name_model, name_data, epoch), epochs
    )
    ## Plot
    padding_figure = (1, 5, 1, 1)
    size_figure = (400, 250)
    size_title = 20
    size_label = 20
    size_tick_label = 16
    title = "Hat Matrix Spectrum ($(name_model), $(name_data))"
    label_y = "Eigenvalue Distribution"
    label_x = "Epoch"
    ticks_x = (1:length(epochs), map(e -> string(e), epochs))
    ##
    if name_data == :CE
        lims_y = (4.0f-4, 4.0f3)
    elseif name_data == :NS
        lims_y = (4.0f-4, 4.0f3)
    end
    ##
    cats = vec(
        stack(map(i -> i * ones(length(eigvals_array[i])), 1:length(epochs)))
    )
    vals = vec(stack(eigvals_array))
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            ylabel=label_y,
            xlabel=label_x,
            ylabelsize=size_label,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            yscale=log10,
            xticks=ticks_x,
            xminorticksvisible=false,
        )
        ylims!(ax, lims_y)
        boxplot!(ax, cats, vals; show_outliers=true, show_notch=true)
        return current_figure()
    end
    ##
    path_save = plotsdir("dynamics/$(name_data)/$(name_model)/eig_dynamics.pdf")
    wsave(path_save, fig)
    ##
    return fig
end
##
function get_eigvals_Q(name_model::Symbol, name_data::Symbol, epoch::Int)
    ##
    dir_load = projectdir("results/Eigen/$(name_data)/$(name_model)/epoch/")
    paths_load = PDEHats.find_files_by_suffix(dir_load, ".jld2")
    filter!(p -> occursin("epoch_$(epoch)_", p), paths_load)
    eigvals_array = map(paths_load) do p
        vals = load(p)["vals"]
        return vals
    end
    eigvals = vec(stack(eigvals_array))
    ##
    return eigvals
end
##
function plot_eig_ratio_Q(name_model::Symbol, name_data::Symbol)
    ##
    if name_data == :CE
        if name_model == :ViT
            epochs = [1, 25, 50, 75, 100, 105, 120, 135, 150]
        elseif name_model == :UNet
            epochs = [1, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
        end
    elseif name_data == :NS
        epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    end
    ##
    eigvals_array = map(
        epoch -> get_eigvals_ratio_Q(name_model, name_data, epoch), epochs
    )
    ## Plot
    padding_figure = (1, 5, 1, 1)
    size_figure = (400, 250)
    size_title = 20
    size_label = 20
    size_tick_label = 16
    title = "Hat Matrix Spectrum ($(name_model), $(name_data))"
    label_y = "Eigenvalue Ratio"
    label_x = "Epoch"
    ticks_x = (1:length(epochs), map(e -> string(e), epochs))
    ##
    cats = vec(
        stack(map(i -> i * ones(length(eigvals_array[i])), 1:length(epochs)))
    )
    vals = vec(stack(eigvals_array))
    if name_data == :CE
        lims_y = (8.0f1, 1.0f6)
    elseif name_data == :NS
        lims_y = (8.0f1, 1.0f6)
    end
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            ylabel=label_y,
            xlabel=label_x,
            ylabelsize=size_label,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            yscale=log10,
            xticks=ticks_x,
            xminorticksvisible=false,
        )
        ylims!(ax, lims_y)
        boxplot!(ax, cats, vals; show_outliers=true, show_notch=false)
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "dynamics/$(name_data)/$(name_model)/eig_ratio_dynamics.pdf"
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
function get_eigvals_ratio_Q(name_model::Symbol, name_data::Symbol, epoch::Int)
    ##
    dir_load = projectdir("results/Eigen/$(name_data)/$(name_model)/epoch/")
    paths_load = PDEHats.find_files_by_suffix(dir_load, ".jld2")
    filter!(p -> occursin("epoch_$(epoch)_", p), paths_load)
    eigvals_array = map(paths_load) do p
        vals_max = maximum(load(p)["vals"])
        vals_min = minimum(load(p)["vals"])
        return vals_max / vals_min
    end
    eigvals = vec(stack(eigvals_array))
    ##
    return eigvals
end
