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
##
include(projectdir("scripts/HatMatrix/obs_batch.jl"))
include(projectdir("scripts/HatMatrix/err_eqv.jl"))
##
function plot_extrapolate()
    plot_extrapolate(:CE)
    plot_extrapolate(:NS)
    return nothing
end
function plot_extrapolate(name_data::Symbol)
    ## UNet
    path_save_UNet = projectdir("results/Benchmark/$(name_data)/UNet/")
    name_save_UNet = "extrapolate_errs"
    dict_UNet, path_UNet =
        produce_or_load((;), path_save_UNet; filename=name_save_UNet) do _
            vals_UNet, dt_vals_UNet, rel_dt_vals_UNet, T_max = plot_extrapolate(
                :UNet, name_data
            )
            dict_result = Dict(
                "vals" => vals_UNet,
                "T_max" => T_max,
                "dt_vals" => dt_vals_UNet,
                "rel_dt_vals" => rel_dt_vals_UNet,
            )
            return dict_result
        end
    vals_UNet = dict_UNet["vals"]
    dt_vals_UNet = dict_UNet["dt_vals"]
    rel_dt_vals_UNet = dict_UNet["rel_dt_vals"]
    T_max = dict_UNet["T_max"]
    ## ViT
    path_save_ViT = projectdir("results/Benchmark/$(name_data)/ViT/")
    name_save_ViT = "extrapolate_errs"
    dict_ViT, path_ViT =
        produce_or_load((;), path_save_ViT; filename=name_save_ViT) do _
            vals_ViT, dt_vals_ViT, rel_dt_vals_ViT, T_max = plot_extrapolate(
                :ViT, name_data
            )
            dict_result = Dict(
                "vals" => vals_ViT,
                "T_max" => T_max,
                "dt_vals" => dt_vals_ViT,
                "rel_dt_vals" => rel_dt_vals_ViT,
            )
            return dict_result
        end
    vals_ViT = dict_ViT["vals"]
    dt_vals_ViT = dict_ViT["dt_vals"]
    rel_dt_vals_ViT = dict_ViT["rel_dt_vals"]
    ##
    title = "Inference Performance ($(name_data))"
    label_x = "Extrapolation Time"
    label_y = "Test Error (SMSE)"
    size_title = 20
    size_label = 18
    size_tick_label = 16
    size_figure = (400, 250)
    size_marker = 8
    width_line = 1
    width_whisker = 10
    padding_figure = (1, 5, 5, 1)
    gap_row = 1
    size_label_legend = 18
    range_t = 17:21
    ##
    U1 = map(v -> v[1], vals_UNet)
    U2 = map(v -> v[2], vals_UNet)
    U3 = map(v -> v[3], vals_UNet)
    V1 = map(v -> v[1], vals_ViT)
    V2 = map(v -> v[2], vals_ViT)
    V3 = map(v -> v[3], vals_ViT)
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
        scatter!(ax, range_t, U2; label="UNet", markersize=size_marker)
        rangebars!(
            ax,
            range_t,
            U1,
            U3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(ax, range_t, V2; label="ViT", markersize=size_marker)
        rangebars!(
            ax,
            range_t,
            V1,
            V3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(
            ax; position=:lt, labelsize=size_label_legend, rowgap=gap_row
        )
        return current_figure()
    end
    ##
    wsave(plotsdir("Benchmark/$(name_data)/extrapolate.pdf"), fig)
    ##
    range_t = 17:20
    label_y = "Test Error Slope (SMSE)"
    U1 = map(v -> v[1], dt_vals_UNet)
    U2 = map(v -> v[2], dt_vals_UNet)
    U3 = map(v -> v[3], dt_vals_UNet)
    V1 = map(v -> v[1], dt_vals_ViT)
    V2 = map(v -> v[2], dt_vals_ViT)
    V3 = map(v -> v[3], dt_vals_ViT)
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
        scatter!(ax, range_t, U2; label="UNet", markersize=size_marker)
        rangebars!(
            ax,
            range_t,
            U1,
            U3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(ax, range_t, V2; label="ViT", markersize=size_marker)
        rangebars!(
            ax,
            range_t,
            V1,
            V3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(
            ax; position=:lt, labelsize=size_label_legend, rowgap=gap_row
        )
        return current_figure()
    end
    ##
    wsave(plotsdir("Benchmark/$(name_data)/extrapolate_slope.pdf"), fig)
    ##
    range_t = 17:20
    label_y = "Relative Error Change (SMSE)"
    U1 = map(v -> v[1], rel_dt_vals_UNet)
    U2 = map(v -> v[2], rel_dt_vals_UNet)
    U3 = map(v -> v[3], rel_dt_vals_UNet)
    V1 = map(v -> v[1], rel_dt_vals_ViT)
    V2 = map(v -> v[2], rel_dt_vals_ViT)
    V3 = map(v -> v[3], rel_dt_vals_ViT)
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
        scatter!(ax, range_t, U2; label="UNet", markersize=size_marker)
        rangebars!(
            ax,
            range_t,
            U1,
            U3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(ax, range_t, V2; label="ViT", markersize=size_marker)
        rangebars!(
            ax,
            range_t,
            V1,
            V3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(
            ax; position=:lb, labelsize=size_label_legend, rowgap=gap_row
        )
        return current_figure()
    end
    ##
    wsave(plotsdir("Benchmark/$(name_data)/extrapolate_slope_rel.pdf"), fig)
    ##
    return fig
end
##
function plot_extrapolate(name_model::Symbol, name_data::Symbol)
    ##
    dev = gpu_device()
    if name_data == :CE
        epoch = 150
        seeds = (10, 35, 42)
    elseif name_data == :NS
        epoch = 100
        seeds = (10, 42)
    end
    ##
    ratio_train = 0.65f0
    ratio_val = 0.05f0
    lambda = 1.0f-4
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    ##
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    ##
    t1 = 16
    dT = 5
    T_max = t1 + dT
    errs_array =
        map(Iterators.product(1:dT, idx_NTs, seeds)) do (dt, idx_NT, seed)
            t2 = t1 + dt
            #
            dir_load = projectdir(
                "results/Train/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)/",
            )
            # CKPT
            keys_ckpt_to_load = ("chs", "st", "ps")
            path_ckpt = dir_load * "checkpoint_$(epoch).jld2"
            ckpt = PDEHats.load_keys_jld2(path_ckpt, keys_ckpt_to_load)
            chs = ckpt["chs"]
            st = ckpt["st"] |> dev
            ps = ckpt["ps"]
            ps = ComponentArray(ps) |> dev
            # Model
            model = PDEHats.get_model(chs, name_model, name_data)
            #
            dir_load = projectdir(
                "results/Train/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)/",
            )
            obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
            (input, target) = obs
            initial = selectdim(input, 4, 1:1)
            trajectory = cat(initial, target; dims=4)
            @assert trajectory[:, :, :, 1:1, :] == initial
            trajectory_t1 = selectdim(trajectory, 4, t1:t1) |> dev
            trajectory_t2 = selectdim(trajectory, 4, t2:t2) |> dev
            pred_t2 = PDEHats.rollout(model, trajectory_t1, ps, st, dt)
            errs = dropdims(
                loss_smse(trajectory_t1, trajectory_t2, pred_t2); dims=1
            )
            return errs |> cpu_device()
        end
    ## [class, time, batch, seed]
    errs_r_p = stack(errs_array)
    errs_r = permutedims(errs_r_p, (2, 1, 3, 4))
    errs = reshape(errs_r, (dT, length(seeds) * length(idx_NTs) * 3))
    ##
    errs_dt = map(1:dT) do dt
        err_dt = errs[dt, :]
        return quantile(err_dt)[2:4]
    end
    diff_errs = diff(errs; dims=1)
    dt_errs_dt = map(1:(dT - 1)) do dt
        diff_err_dt = diff_errs[dt, :]
        return quantile(diff_err_dt)[2:4]
    end
    rel_dt_errs_dt = map(1:(dT - 1)) do dt
        err_dt = errs[dt, :]
        diff_err_dt = diff_errs[dt, :]
        return quantile(diff_err_dt ./ err_dt)[2:4]
    end
    ##
    return errs_dt, dt_errs_dt, rel_dt_errs_dt, T_max
end
