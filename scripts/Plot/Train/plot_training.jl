##
using DrWatson
@quickactivate :PDEHats
#
using Lux, LuxCUDA
using CairoMakie, MakiePublication
using Statistics
##
function plot_training()
    ##
    name_model_array=["UNet", "ViT"]
    seed_array=[35, 42, 10]
    ##
    losses_model_seed_array = map(name_model_array) do name_model
        losses_seed_array = map(seed_array) do seed
            dir_load_ckpt_array = readdir(
                projectdir("results/Train/$(name_model)/seed_$(seed)/");
                join=true,
            )
            losses_array = map(dir_load_ckpt_array) do dir_load_ckpt
                path_val = only(
                    PDEHats.find_files_by_suffix(
                        dir_load_ckpt, "val-fn=loss_mse.jld2"
                    ),
                )
                losses = load(path_val)["history_val"]
                return losses
            end
            losses_seed = vcat(losses_array...)
            return losses_seed
        end
        losses_seeds = stack(losses_seed_array)
        losses_min = dropdims(minimum(losses_seeds; dims=2); dims=2)
        losses_median = dropdims(median(losses_seeds; dims=2); dims=2)
        losses_max = dropdims(maximum(losses_seeds; dims=2); dims=2)
        return (losses_min, losses_median, losses_max)
    end
    ##
    title = "Optimization Curve (Validation Data)"
    label_1 = name_model_array[1]
    label_2 = name_model_array[2]
    label_x = "Epoch"
    label_y = "SMSE"
    size_title = 18
    size_label=16
    size_tick_label=14
    size_figure = (400, 250)
    size_marker = 8
    width_line = 1
    width_whisker = 10
    padding_figure=(1, 1, 5, 1)
    #
    (min_1, med_1, max_1) = losses_model_seed_array[1]
    (min_2, med_2, max_2) = losses_model_seed_array[2]
    #
    epochs_1 = 1:length(med_1)
    epochs_2 = 1:length(med_2)
    #
    step_val = 5
    steps_val_1 = union(1, step_val:step_val:length(epochs_1), length(epochs_1))
    steps_val_2 = union(1, step_val:step_val:length(epochs_2), length(epochs_2))
    #
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
    wsave(projectdir("results/Train/opt_curve.pdf"), fig)
    ##
    return nothing
end
