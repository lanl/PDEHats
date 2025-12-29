##
function plot_err_eqv_Z()
    plot_err_eqv_ZX()
    plot_err_eqv_ZY()
    return nothing
end
function plot_err_eqv_ZX()
    ##
    name_data = :CE
    loss_fn = :loss_smse
    dir_save = plotsdir("Eqv/$(name_data)/$(loss_fn)/D4/err_eqv/")
    L = 128
    ##
    ket = "ket_C_smse"
    ket_gs = map(0:(L - 1)) do dx
        if dx == 0
            return "g_identity"
        else
            return "g_shift_x_$(dx)_y_0"
        end
    end
    ## UNet
    name_model = :UNet
    results_paths_UNet = PDEHats.find_files(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv",
        ".jld2",
    )
    results_UNet = map(ket_gs) do ket_g
        results_paths_g = filter(
            p -> occursin("$(ket_g).jld2", p), results_paths_UNet
        )
        @assert length(results_paths_g) == 18
        results_g = map(p -> load(p)["err_eqv"], results_paths_g)
        return stack(results_g)
    end
    ## ViT
    name_model = :ViT
    results_paths_ViT = PDEHats.find_files(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv",
        ".jld2",
    )
    results_ViT = map(ket_gs) do ket_g
        results_paths_g = filter(
            p -> occursin("$(ket_g).jld2", p), results_paths_ViT
        )
        @assert length(results_paths_g) == 18
        results_g = map(p -> load(p)["err_eqv"], results_paths_g)
        return stack(results_g)
    end
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    label_x = "Translation (Horizontal)"
    label_y = "SMSE"
    size_figure = (800, 450)
    size_patch = (8, 8)
    gap_row = 4
    size_legend = 32
    size_title = 40
    size_label = 32
    size_tick_label = 32
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    ## Time
    quants_UNet = map(1:L) do l
        return quantile(vec(results_UNet[l]))
    end
    q2_UNet = circshift(map(q -> q[2], quants_UNet), 63)
    q3_UNet = circshift(map(q -> q[3], quants_UNet), 63)
    q4_UNet = circshift(map(q -> q[4], quants_UNet), 63)
    quants_ViT = map(1:L) do l
        return quantile(vec(results_ViT[l]))
    end
    q2_ViT = circshift(map(q -> q[2], quants_ViT), 63)
    q3_ViT = circshift(map(q -> q[3], quants_ViT), 63)
    q4_ViT = circshift(map(q -> q[4], quants_ViT), 63)
    title = "Equivariance Error"
    ##
    range_x = 1:128
    skipper = 3
    range_x = range_x[1:skipper:end]
    q2_UNet = q2_UNet[1:skipper:end]
    q3_UNet = q3_UNet[1:skipper:end]
    q4_UNet = q4_UNet[1:skipper:end]
    q2_ViT = q2_ViT[1:skipper:end]
    q3_ViT = q3_ViT[1:skipper:end]
    q4_ViT = q4_ViT[1:skipper:end]
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            ylabelsize=size_label,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            xticks=xticks,
        )
        scatterlines!(
            ax, range_x, q3_UNet; label="UNet", linewidth=2, markersize=14
        )
        rangebars!(ax, range_x, q4_UNet, q2_UNet; whiskerwidth=4)
        scatterlines!(
            ax,
            range_x,
            q3_ViT;
            label="ViT",
            marker=:rect,
            linewidth=2,
            markersize=14,
        )
        rangebars!(ax, range_x, q4_ViT, q2_ViT; whiskerwidth=4)
        Legend(
            fig[1, 2],
            ax;
            patchsize=size_patch,
            rowgap=gap_row,
            labelsize=size_legend,
        )
        return current_figure()
    end
    ##
    # wsave(projectdir(dir_save * "time_$t.pdf"), fig)
    return fig
end
##
function plot_err_eqv_ZY()
    ##
    name_data = :CE
    loss_fn = :loss_smse
    dir_save = plotsdir("Eqv/$(name_data)/$(loss_fn)/D4/err_eqv/")
    L = 128
    ##
    ket = "ket_C_smse"
    ket_gs = map(0:(L - 1)) do dy
        if dy == 0
            return "g_identity"
        else
            return "g_shift_x_0_y_$(dy)"
        end
    end
    ## UNet
    name_model = :UNet
    results_paths_UNet = PDEHats.find_files(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv",
        ".jld2",
    )
    results_UNet = map(ket_gs) do ket_g
        results_paths_g = filter(
            p -> occursin("$(ket_g).jld2", p), results_paths_UNet
        )
        results_g = map(p -> load(p)["err_eqv"], results_paths_g)
        return stack(results_g)
    end
    ## ViT
    name_model = :ViT
    results_paths_ViT = PDEHats.find_files(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv",
        ".jld2",
    )
    results_ViT = map(ket_gs) do ket_g
        results_paths_g = filter(
            p -> occursin("$(ket_g).jld2", p), results_paths_ViT
        )
        results_g = map(p -> load(p)["err_eqv"], results_paths_g)
        return stack(results_g)
    end
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    label_x = "Translation (Vertical)"
    label_y = "SMSE"
    size_figure = (800, 450)
    size_patch = (8, 8)
    gap_row = 4
    size_legend = 32
    size_title = 40
    size_label = 32
    size_tick_label = 32
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    ## Time
    quants_UNet = map(1:L) do l
        return quantile(results_UNet[l])
    end
    q2_UNet = circshift(map(q -> q[2], quants_UNet), 63)
    q3_UNet = circshift(map(q -> q[3], quants_UNet), 63)
    q4_UNet = circshift(map(q -> q[4], quants_UNet), 63)
    quants_ViT = map(1:L) do l
        return quantile(results_ViT[l])
    end
    q2_ViT = circshift(map(q -> q[2], quants_ViT), 63)
    q3_ViT = circshift(map(q -> q[3], quants_ViT), 63)
    q4_ViT = circshift(map(q -> q[4], quants_ViT), 63)
    title = "Equivariance Error"
    ##
    range_x = 1:128
    skipper = 3
    range_x = range_x[1:skipper:end]
    q2_UNet = q2_UNet[1:skipper:end]
    q3_UNet = q3_UNet[1:skipper:end]
    q4_UNet = q4_UNet[1:skipper:end]
    q2_ViT = q2_ViT[1:skipper:end]
    q3_ViT = q3_ViT[1:skipper:end]
    q4_ViT = q4_ViT[1:skipper:end]
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            ylabelsize=size_label,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            xticks=xticks,
        )
        scatterlines!(
            ax, range_x, q3_UNet; label="UNet", linewidth=2, markersize=14
        )
        rangebars!(ax, range_x, q4_UNet, q2_UNet; whiskerwidth=4)
        scatterlines!(
            ax,
            range_x,
            q3_ViT;
            label="ViT",
            marker=:rect,
            linewidth=2,
            markersize=14,
        )
        rangebars!(ax, range_x, q4_ViT, q2_ViT; whiskerwidth=4)
        Legend(
            fig[1, 2],
            ax;
            patchsize=size_patch,
            rowgap=gap_row,
            labelsize=size_legend,
        )
        return current_figure()
    end
    ##
    # wsave(projectdir(dir_save * "time_$t.pdf"), fig)
    return fig
end
##
