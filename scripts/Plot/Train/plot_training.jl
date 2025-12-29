##
using DrWatson
@quickactivate :PDEHats
#
using Lux, LuxCUDA
using CairoMakie, MakiePublication
using Statistics
##
function plot_training_CE()
    ##
    seeds = (35, 42, 10)
    epochs = 150
    ## ViT
    val_paths_ViT = PDEHats.find_files(
        projectdir("results/Train/CE/ViT"), "val-fn=loss_mse_scaled", ".jld2"
    )
    losses_ViT = map(seeds) do seed
        val_paths_seed = filter(p -> occursin("seed_$(seed)", p), val_paths_ViT)
        vals_1 = load(
            only(filter(p -> occursin("epoch_1_to_100", p), val_paths_seed))
        )["history_val"]
        vals_2 = load(
            only(filter(p -> occursin("epoch_101_to_130", p), val_paths_seed)),
        )["history_val"]
        vals_3 = load(
            only(filter(p -> occursin("epoch_131_to_150", p), val_paths_seed)),
        )["history_val"]
        vals = vcat(vals_1, vals_2, vals_3)
        return vals
    end
    min_1 = dropdims(minimum(stack(losses_ViT); dims=2); dims=2)
    med_1 = dropdims(median(stack(losses_ViT); dims=2); dims=2)
    max_1 = dropdims(maximum(stack(losses_ViT); dims=2); dims=2)
    ## UNet
    val_paths_UNet = PDEHats.find_files(
        projectdir("results/Train/CE/UNet"), "val-fn=loss_mse_scaled", ".jld2"
    )
    losses_UNet = map(seeds) do seed
        val_paths_seed = filter(p -> occursin("seed_$(seed)", p), val_paths_UNet)
        vals = load(
            only(filter(p -> occursin("epoch_1_to_150", p), val_paths_seed))
        )["history_val"]
        return vals
    end
    min_2 = dropdims(minimum(stack(losses_UNet); dims=2); dims=2)
    med_2 = dropdims(median(stack(losses_UNet); dims=2); dims=2)
    max_2 = dropdims(maximum(stack(losses_UNet); dims=2); dims=2)
    ##
    title = "Optimization Curve (Test Data)"
    label_1 = "ViT"
    label_2 = "UNet"
    label_x = "Epoch"
    label_y = "SMSE"
    size_title = 18
    size_label = 16
    size_tick_label = 14
    size_figure = (400, 250)
    size_marker = 8
    width_line = 1
    width_whisker = 10
    padding_figure = (1, 1, 5, 1)
    #
    epochs_1 = 1:length(med_1)
    epochs_2 = 1:length(med_2)
    #
    step_val = 5
    steps_val_1 = union(1, step_val:step_val:length(epochs_1), length(epochs_1))
    steps_val_2 = union(1, step_val:step_val:length(epochs_2), length(epochs_2))
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
        scatter!(
            ax,
            epochs_1[steps_val_1],
            med_1[steps_val_1];
            label=label_1,
            markersize=size_marker,
        )
        rangebars!(
            ax,
            epochs_1[steps_val_1],
            min_1[steps_val_1],
            max_1[steps_val_1];
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(
            ax,
            epochs_2[steps_val_2],
            med_2[steps_val_2];
            label=label_2,
            markersize=size_marker,
        )
        rangebars!(
            ax,
            epochs_2[steps_val_2],
            min_2[steps_val_2],
            max_2[steps_val_2];
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(ax; position=:rt, labelsize=size_label, rowgap=1)
        return current_figure()
    end
    # wsave(projectdir(dir_saved * "Train/opt_curve.pdf"), fig)
    ##
    return fig
end
