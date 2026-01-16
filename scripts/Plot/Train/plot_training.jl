##
using DrWatson
@quickactivate :PDEHats
#
using Lux
using CairoMakie, MakiePublication
using Statistics
##
function plot_training()
    fig = plot_training_CE()
    fig = plot_training_NS()
    return nothing
end
function plot_training_CE()
    ##
    name_data = :CE
    seeds = (35, 42, 10)
    epochs = 150
    skipper = 5
    range_e = vcat(collect(1:epochs)[1:skipper:end], 150)
    ## ViT
    val_paths_ViT = PDEHats.find_files(
        projectdir("results/Train/$(name_data)/ViT"),
        "val-fn=loss_mse_scaled",
        ".jld2",
    )
    losses_ViT = map(seeds) do seed
        val_paths_seed = filter(p -> occursin("seed_$(seed)", p), val_paths_ViT)
        vals_1 = load(
            only(filter(p -> occursin("epoch_1_to_100", p), val_paths_seed))
        )["history_val"]
        vals_2 = load(
            only(filter(p -> occursin("epoch_101_to_130", p), val_paths_seed)),
        )["history_val"][101:130]
        vals_3 = load(
            only(filter(p -> occursin("epoch_131_to_150", p), val_paths_seed)),
        )["history_val"]
        vals = vcat(vals_1, vals_2, vals_3)
        return vals
    end
    ##
    Vs = map(1:epochs) do epoch
        V = collect(map(l -> l[epoch], losses_ViT))
        return quantile(V)[2:4]
    end
    V1 = map(v -> v[1], Vs)[range_e]
    V2 = map(v -> v[2], Vs)[range_e]
    V3 = map(v -> v[3], Vs)[range_e]
    ## UNet
    val_paths_UNet = PDEHats.find_files(
        projectdir("results/Train/$(name_data)/UNet"),
        "val-fn=loss_mse_scaled",
        ".jld2",
    )
    losses_UNet = map(seeds) do seed
        val_paths_seed = filter(p -> occursin("seed_$(seed)", p), val_paths_UNet)
        vals = load(
            only(filter(p -> occursin("epoch_1_to_150", p), val_paths_seed))
        )["history_val"]
        return vals
    end
    Us = map(1:epochs) do epoch
        U = collect(map(l -> l[epoch], losses_UNet))
        return quantile(U)[2:4]
    end
    U1 = map(v -> v[1], Us)[range_e]
    U2 = map(v -> v[2], Us)[range_e]
    U3 = map(v -> v[3], Us)[range_e]
    ## Plotting
    padding_figure = (1, 5, 5, 1)
    size_figure = (400, 250)
    title = "Optimization Curve ($(name_data))"
    label_x = "Epoch"
    label_y = "Test Error (SMSE)"
    size_title = 22
    size_label = 20
    size_label_legend = 20
    size_tick_label = 18
    size_marker = 10
    width_line = 1
    width_whisker = 12
    gap_row = 0.25
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            titlesize=size_title,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            yscale=log10,
        )
        scatter!(ax, range_e, U2; label="UNet", markersize=size_marker)
        rangebars!(
            ax,
            range_e,
            U1,
            U3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(ax, range_e, V2; label="ViT", markersize=size_marker)
        rangebars!(
            ax,
            range_e,
            V1,
            V3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(
            ax; position=:rt, labelsize=size_label_legend, rowgap=gap_row
        )
        return current_figure()
    end
    wsave(plotsdir("Train/$(name_data)/opt_curve.pdf"), fig)
    ##
    return fig
end
function plot_training_NS()
    ##
    name_data = :NS
    epochs = 100
    skipper = 5
    range_e = vcat(collect(1:epochs)[1:skipper:end], 100)
    ## ViT
    seeds = (10, 42)
    val_paths_ViT = PDEHats.find_files(
        projectdir("results/Train/$(name_data)/ViT"),
        "val-fn=loss_mse_scaled",
        ".jld2",
    )
    losses_ViT = map(seeds) do seed
        val_paths_seed = filter(p -> occursin("seed_$(seed)", p), val_paths_ViT)
        vals = load(
            only(filter(p -> occursin("epoch_1_to_100", p), val_paths_seed))
        )["history_val"]
        return vals
    end
    Vs = map(1:epochs) do epoch
        V = collect(map(l -> l[epoch], losses_ViT))
        return quantile(V)[2:4]
    end
    V1 = map(v -> v[1], Vs)[range_e]
    V2 = map(v -> v[2], Vs)[range_e]
    V3 = map(v -> v[3], Vs)[range_e]
    ## UNet
    seeds = (10, 35, 42)
    val_paths_UNet = PDEHats.find_files(
        projectdir("results/Train/$(name_data)/UNet"),
        "val-fn=loss_mse_scaled",
        ".jld2",
    )
    losses_UNet = map(seeds) do seed
        val_paths_seed = filter(p -> occursin("seed_$(seed)", p), val_paths_UNet)
        vals = load(
            only(filter(p -> occursin("epoch_1_to_100", p), val_paths_seed))
        )["history_val"]
        return vals
    end
    Us = map(1:epochs) do epoch
        U = collect(map(l -> l[epoch], losses_UNet))
        return quantile(U)[2:4]
    end
    U1 = map(v -> v[1], Us)[range_e]
    U2 = map(v -> v[2], Us)[range_e]
    U3 = map(v -> v[3], Us)[range_e]
    ##
    ## Plotting
    padding_figure = (1, 5, 5, 1)
    size_figure = (400, 250)
    title = "Optimization Curve ($(name_data))"
    label_x = "Epoch"
    label_y = "Test Error (SMSE)"
    size_title = 22
    size_label = 20
    size_label_legend = 20
    size_tick_label = 18
    size_marker = 10
    width_line = 1
    width_whisker = 12
    gap_row = 0.25
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            titlesize=size_title,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            yscale=log10,
        )
        scatter!(ax, range_e, U2; label="UNet", markersize=size_marker)
        rangebars!(
            ax,
            range_e,
            U1,
            U3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(ax, range_e, V2; label="ViT", markersize=size_marker)
        rangebars!(
            ax,
            range_e,
            V1,
            V3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(
            ax; position=:rt, labelsize=size_label_legend, rowgap=gap_row
        )
        axislegend(
            ax; position=:rt, labelsize=size_label_legend, rowgap=gap_row
        )
        return current_figure()
    end
    wsave(plotsdir("Train/$(name_data)/opt_curve.pdf"), fig)
    ##
    return fig
end
