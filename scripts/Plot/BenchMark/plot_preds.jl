##
using DrWatson
@quickactivate :PDEHats
#
using Lux, LuxCUDA
using ComponentArrays
using MLUtils
using Random
using Statistics
using CairoMakie, MakiePublication
#
include(projectdir("scripts/HatMatrix/obs_batch.jl"))
##
function plot_preds()
    ##
    name_models = (:UNet, :ViT)
    name_datas = (:CE, :NS)
    for name_model in name_models
        for name_data in name_datas
            try
                plot_preds(name_model, name_data)
            catch e
                println(e)
            end
        end
    end
    ##
    return nothing
end
##
function plot_preds(name_model::Symbol, name_data::Symbol)
    ##
    dev = gpu_device()
    ##
    if name_data == :CE
        epoch = 150
        classes_data = ("RP", "CRP", "RPUI")
    elseif name_data == :NS
        epoch = 100
        classes_data = ("BB", "Gauss", "Sines")
    end
    ##
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    ##
    ratio_train = 0.65f0
    ratio_val = 0.05f0
    lambda = 1.0f-4
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    T_max = 21
    ##
    for (seed, idx_NT) in Iterators.product(seeds, idx_NTs)
        ##
        idx_rp = idx_NT.idx_rp
        idx_crp = idx_NT.idx_crp
        idx_rpui = idx_NT.idx_rpui
        dir_batch = savename(
            (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
        )
        ## CKPT
        dir_load = projectdir(
            "results/Train/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)/",
        )
        keys_ckpt_to_load = ("chs", "st", "ps")
        path_ckpt = dir_load * "checkpoint_$(epoch).jld2"
        ckpt = PDEHats.load_keys_jld2(path_ckpt, keys_ckpt_to_load)
        chs = ckpt["chs"]
        st = ckpt["st"] |> dev
        ps = ckpt["ps"]
        ps = ComponentArray(ps) |> dev
        ## Model
        model = PDEHats.get_model(chs, name_model, name_data)
        obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
        (input_cpu, target) = obs
        input = input_cpu |> gpu_device()
        pred = PDEHats.rollout(model, input, ps, st) |> cpu_device()
        (Lx, Ly, F, T, B) = size(pred)
        ##
        titles = ("Prediction", "Target", "Difference")
        if F == 4
            labels_y = ("Mass", "Momentum-x", "Momentum-y", "Energy")
        elseif F == 2
            labels_y = ("Velocity-x", "Velocity-y")
        end
        ##
        for b in 1:B
            ##
            tuple_1 = map(Tuple(1:F)) do f
                pred_fb = pred[:, :, f, :, b]
                return pred_fb
            end
            tuple_2 = map(Tuple(1:F)) do f
                target_fb = target[:, :, f, :, b]
                return target_fb
            end
            tuple_3 = map(Tuple(1:F)) do f
                pred_fb = pred[:, :, f, :, b]
                target_fb = target[:, :, f, :, b]
                return pred_fb .- target_fb
            end
            ##
            path_save = plotsdir(
                "Benchmark/Preds/$(name_data)/$(name_model)/seed_$(seed)/$(dir_batch)/$(classes_data[b])/rollout",
            )
            supertitle = "Rollout ($(name_model), $(classes_data[b]), Time Step: "
            animate_heatmaps_4x3(
                tuple_1,
                tuple_2,
                tuple_3,
                supertitle,
                titles,
                labels_y;
                path_save=path_save,
            )
            ##
        end
    end
    ##
    return nothing
end
function animate_heatmaps_4x3(
    tuple_1::NTuple{4,AbstractArray{Float32,3}},
    tuple_2::NTuple{4,AbstractArray{Float32,3}},
    tuple_3::NTuple{4,AbstractArray{Float32,3}},
    supertitle::AbstractString,
    titles::NTuple{3,AbstractString},
    labels_y::NTuple{4,AbstractString};
    size_supertitle::Int=18,
    size_title::Int=12,
    size_ylabel::Int=12,
    path_save::String=projectdir("dir_save_default/animate_heatmaps_4x3"),
    framerate::Int=3,
    size_fig::NTuple{2,Int}=(310, 400),
    padding_figure::NTuple{4,Int}=(1, 1, 20, 1),
    label_colorbar::Union{Nothing,AbstractString}=nothing,
    size_label_colorbar::Int=10,
    colormap=:balance,
)
    ##
    trajectory_1 = stack(tuple_1; dims=3)
    trajectory_2 = stack(tuple_2; dims=3)
    trajectory_3 = stack(tuple_3; dims=3)
    ## Range: Min
    min_1 = minimum(trajectory_1; dims=(1, 2, 4))
    min_2 = minimum(trajectory_2; dims=(1, 2, 4))
    min_3 = minimum(trajectory_3; dims=(1, 2, 4))
    ranges_color_min = vec(minimum(cat(min_1, min_2, min_3; dims=1); dims=1))
    ## Range: Max
    max_1 = maximum(trajectory_1; dims=(1, 2, 4))
    max_2 = maximum(trajectory_2; dims=(1, 2, 4))
    max_3 = maximum(trajectory_3; dims=(1, 2, 4))
    ranges_color_max = vec(maximum(cat(max_1, max_2, max_3; dims=1); dims=1))
    ##
    range_color_rows = map(1:size(trajectory_1, 3)) do i
        if (i == 1) || (i == 4)
            val = max(abs(ranges_color_min[i]), abs(ranges_color_max[i]))
        elseif (i == 2) || (i == 3)
            val_2 = max(abs(ranges_color_min[2]), abs(ranges_color_max[2]))
            val_3 = max(abs(ranges_color_min[3]), abs(ranges_color_max[3]))
            val = max(val_2, val_3)
        end
        return (-val, val)
    end
    times = 1:size(trajectory_1, 4)
    time_Obs = Observable(times[1])
    ## Density
    density_1 = @lift(trajectory_1[:, :, 1, $time_Obs])
    density_2 = @lift(trajectory_2[:, :, 1, $time_Obs])
    density_3 = @lift(trajectory_3[:, :, 1, $time_Obs])
    ## Momentum-x
    momentum_x_1 = @lift(trajectory_1[:, :, 2, $time_Obs])
    momentum_x_2 = @lift(trajectory_2[:, :, 2, $time_Obs])
    momentum_x_3 = @lift(trajectory_3[:, :, 2, $time_Obs])
    ## Momentum-y
    momentum_y_1 = @lift(trajectory_1[:, :, 3, $time_Obs])
    momentum_y_2 = @lift(trajectory_2[:, :, 3, $time_Obs])
    momentum_y_3 = @lift(trajectory_3[:, :, 3, $time_Obs])
    ## Energy
    energy_1 = @lift(trajectory_1[:, :, 4, $time_Obs])
    energy_2 = @lift(trajectory_2[:, :, 4, $time_Obs])
    energy_3 = @lift(trajectory_3[:, :, 4, $time_Obs])
    ##
    supertitle_Obs = @lift(supertitle * string($time_Obs) * ")")
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        Label(fig[0, 1:3], supertitle_Obs; fontsize=size_supertitle)
        # Density
        ax_11 = Makie.Axis(
            fig[1, 1];
            title=titles[1],
            titlesize=size_title,
            ylabel=labels_y[1],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_12 = Makie.Axis(
            fig[1, 2];
            title=titles[2],
            titlesize=size_title,
            aspect=DataAspect(),
        )
        ax_13 = Makie.Axis(
            fig[1, 3];
            title=titles[3],
            titlesize=size_title,
            aspect=DataAspect(),
        )
        #
        hidedecorations!(ax_11; label=false)
        hidedecorations!(ax_12; label=false)
        hidedecorations!(ax_13; label=false)
        #
        hm_11 = heatmap!(
            ax_11,
            density_1;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        hm_12 = heatmap!(
            ax_12,
            density_2;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        hm_13 = heatmap!(
            ax_13,
            density_3;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        #
        cb_1 = Colorbar(
            fig[1, 4]; colorrange=range_color_rows[1], colormap=colormap
        )
        # Velocity-x
        ax_21 = Makie.Axis(
            fig[2, 1];
            ylabel=labels_y[2],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_22 = Makie.Axis(fig[2, 2]; aspect=DataAspect())
        ax_23 = Makie.Axis(fig[2, 3]; aspect=DataAspect())
        #
        hidedecorations!(ax_21; label=false)
        hidedecorations!(ax_22; label=false)
        hidedecorations!(ax_23; label=false)
        #
        hm_21 = heatmap!(
            ax_21,
            momentum_x_1;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        hm_22 = heatmap!(
            ax_22,
            momentum_x_2;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        hm_23 = heatmap!(
            ax_23,
            momentum_x_3;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        #
        cb_2 = Colorbar(
            fig[2, 4]; colorrange=range_color_rows[2], colormap=colormap
        )
        # Velocity-y
        ax_31 = Makie.Axis(
            fig[3, 1];
            ylabel=labels_y[3],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_32 = Makie.Axis(fig[3, 2]; aspect=DataAspect())
        ax_33 = Makie.Axis(fig[3, 3]; aspect=DataAspect())
        #
        hidedecorations!(ax_31; label=false)
        hidedecorations!(ax_32; label=false)
        hidedecorations!(ax_33; label=false)
        #
        hm_31 = heatmap!(
            ax_31,
            momentum_y_1;
            colorrange=range_color_rows[3],
            colormap=colormap,
        )
        hm_32 = heatmap!(
            ax_32,
            momentum_y_2;
            colorrange=range_color_rows[3],
            colormap=colormap,
        )
        hm_33 = heatmap!(
            ax_33,
            momentum_y_3;
            colorrange=range_color_rows[3],
            colormap=colormap,
        )
        #
        cb_3 = Colorbar(
            fig[3, 4]; colorrange=range_color_rows[3], colormap=colormap
        )
        # Energy
        ax_41 = Makie.Axis(
            fig[4, 1];
            ylabel=labels_y[4],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_42 = Makie.Axis(fig[4, 2]; aspect=DataAspect())
        ax_43 = Makie.Axis(fig[4, 3]; aspect=DataAspect())
        #
        hidedecorations!(ax_41; label=false)
        hidedecorations!(ax_42; label=false)
        hidedecorations!(ax_43; label=false)
        #
        hm_41 = heatmap!(
            ax_41,
            energy_1;
            colorrange=range_color_rows[4],
            colormap=colormap,
        )
        hm_42 = heatmap!(
            ax_42,
            energy_2;
            colorrange=range_color_rows[4],
            colormap=colormap,
        )
        hm_43 = heatmap!(
            ax_43,
            energy_3;
            colorrange=range_color_rows[4],
            colormap=colormap,
        )
        #
        cb_4 = Colorbar(
            fig[4, 4]; colorrange=range_color_rows[4], colormap=colormap
        )
        #
        if label_colorbar != nothing
            cb_1.label = label_colorbar
            cb_1.labelsize = size_label_colorbar
            cb_2.label = label_colorbar
            cb_2.labelsize = size_label_colorbar
            cb_3.label = label_colorbar
            cb_3.labelsize = size_label_colorbar
            cb_4.label = label_colorbar
            cb_4.labelsize = size_label_colorbar
        end
        #
        colsize!(fig.layout, 1, Aspect(1, 1))
        colsize!(fig.layout, 2, Aspect(1, 1))
        colsize!(fig.layout, 3, Aspect(1, 1))
        #
        return current_figure()
    end
    ##
    for t in times
        time_Obs[] = t
        wsave(path_save * "_t$(t).pdf", fig)
    end
    CairoMakie.record(
        fig, projectdir(path_save * ".mp4"), times; framerate=framerate
    ) do t
        return time_Obs[] = t
    end
    ##
    return nothing
end
##
function animate_heatmaps_4x3(
    tuple_1::NTuple{2,AbstractArray{Float32,3}},
    tuple_2::NTuple{2,AbstractArray{Float32,3}},
    tuple_3::NTuple{2,AbstractArray{Float32,3}},
    supertitle::AbstractString,
    titles::NTuple{3,AbstractString},
    labels_y::NTuple{2,AbstractString};
    size_supertitle::Int=18,
    size_title::Int=12,
    size_ylabel::Int=12,
    path_save::String=projectdir("dir_save_default/animate_heatmaps_4x3"),
    framerate::Int=3,
    size_fig::NTuple{2,Int}=(310, 200),
    padding_figure::NTuple{4,Int}=(1, 1, 10, 1),
    label_colorbar::Union{Nothing,AbstractString}=nothing,
    size_label_colorbar::Int=10,
    colormap=:balance,
)
    ##
    trajectory_1 = stack(tuple_1; dims=3)
    trajectory_2 = stack(tuple_2; dims=3)
    trajectory_3 = stack(tuple_3; dims=3)
    ## Range: Min
    min_1 = minimum(trajectory_1; dims=(1, 2, 4))
    min_2 = minimum(trajectory_2; dims=(1, 2, 4))
    min_3 = minimum(trajectory_3; dims=(1, 2, 4))
    ranges_color_min = vec(minimum(cat(min_1, min_2, min_3; dims=1); dims=1))
    ## Range: Max
    max_1 = maximum(trajectory_1; dims=(1, 2, 4))
    max_2 = maximum(trajectory_2; dims=(1, 2, 4))
    max_3 = maximum(trajectory_3; dims=(1, 2, 4))
    ranges_color_max = vec(maximum(cat(max_1, max_2, max_3; dims=1); dims=1))
    ##
    range_color_rows = map(1:size(trajectory_1, 3)) do i
        val_1 = max(abs(ranges_color_min[1]), abs(ranges_color_max[1]))
        val_2 = max(abs(ranges_color_min[2]), abs(ranges_color_max[2]))
        val = max(val_1, val_2)
        return (-val, val)
    end
    times = 1:size(trajectory_1, 4)
    time_Obs = Observable(times[1])
    ## Density
    density_1 = @lift(trajectory_1[:, :, 1, $time_Obs])
    density_2 = @lift(trajectory_2[:, :, 1, $time_Obs])
    density_3 = @lift(trajectory_3[:, :, 1, $time_Obs])
    ## Momentum-x
    momentum_x_1 = @lift(trajectory_1[:, :, 2, $time_Obs])
    momentum_x_2 = @lift(trajectory_2[:, :, 2, $time_Obs])
    momentum_x_3 = @lift(trajectory_3[:, :, 2, $time_Obs])
    ##
    supertitle_Obs = @lift(supertitle * string($time_Obs) * ")")
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        Label(fig[0, 1:3], supertitle_Obs; fontsize=size_supertitle)
        # Density
        ax_11 = Makie.Axis(
            fig[1, 1];
            title=titles[1],
            titlesize=size_title,
            ylabel=labels_y[1],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_12 = Makie.Axis(
            fig[1, 2];
            title=titles[2],
            titlesize=size_title,
            aspect=DataAspect(),
        )
        ax_13 = Makie.Axis(
            fig[1, 3];
            title=titles[3],
            titlesize=size_title,
            aspect=DataAspect(),
        )
        #
        hidedecorations!(ax_11; label=false)
        hidedecorations!(ax_12; label=false)
        hidedecorations!(ax_13; label=false)
        #
        hm_11 = heatmap!(
            ax_11,
            density_1;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        hm_12 = heatmap!(
            ax_12,
            density_2;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        hm_13 = heatmap!(
            ax_13,
            density_3;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        #
        cb_1 = Colorbar(
            fig[1, 4]; colorrange=range_color_rows[1], colormap=colormap
        )
        # Velocity-x
        ax_21 = Makie.Axis(
            fig[2, 1];
            ylabel=labels_y[2],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_22 = Makie.Axis(fig[2, 2]; aspect=DataAspect())
        ax_23 = Makie.Axis(fig[2, 3]; aspect=DataAspect())
        #
        hidedecorations!(ax_21; label=false)
        hidedecorations!(ax_22; label=false)
        hidedecorations!(ax_23; label=false)
        #
        hm_21 = heatmap!(
            ax_21,
            momentum_x_1;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        hm_22 = heatmap!(
            ax_22,
            momentum_x_2;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        hm_23 = heatmap!(
            ax_23,
            momentum_x_3;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        #
        cb_2 = Colorbar(
            fig[2, 4]; colorrange=range_color_rows[2], colormap=colormap
        )
        #
        if label_colorbar != nothing
            cb_1.label = label_colorbar
            cb_1.labelsize = size_label_colorbar
            cb_2.label = label_colorbar
            cb_2.labelsize = size_label_colorbar
        end
        #
        colsize!(fig.layout, 1, Aspect(1, 1))
        colsize!(fig.layout, 2, Aspect(1, 1))
        colsize!(fig.layout, 3, Aspect(1, 1))
        #
        return current_figure()
    end
    ##
    for t in times
        time_Obs[] = t
        wsave(path_save * "_t$(t).pdf", fig)
    end
    CairoMakie.record(
        fig, projectdir(path_save * ".mp4"), times; framerate=framerate
    ) do t
        return time_Obs[] = t
    end
    ##
    return nothing
end
