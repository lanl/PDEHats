##
function plot_eqv_lines()
    plot_eqv_lines_x()
    plot_eqv_lines_y()
    return nothing
end
##
function plot_eqv_lines_x()
    ##
    dir_save = "results/HatMatrix/Figures/eqv_err/lines_x/"
    ##
    ket = "ket_C_smse"
    ##
    ket_gs = map(0:127) do dx
        if dx == 0
            return "g_identity"
        else
            return "g_shift_x_$(dx)_y_0"
        end
    end
    ## UNet
    name_model = "UNet"
    results_paths_UNet = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_UNet = map(ket_gs) do ket_g
        results_paths_g = filter(
            p ->
                occursin(ket_g * "_J_" * ket, p) &&
                occursin("errs/mse_scaled/", p),
            results_paths_UNet,
        )
        results_g = map(p -> load(p)[ket_g], results_paths_g)
        return stack(results_g)
    end
    ## ViT
    name_model = "ViT"
    results_paths_ViT = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_ViT = map(ket_gs) do ket_g
        results_paths_g = filter(
            p ->
                occursin(ket_g * "_J_" * ket, p) &&
                occursin("errs/mse_scaled/", p),
            results_paths_ViT,
        )
        results_g = map(p -> load(p)[ket_g], results_paths_g)
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
    L = size(results_ViT, 1)
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    # All
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
    title = "Equivariance Error (All Times)"
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
        ylims!(ax, 0.0, 0.065)
        scatterlines!(ax, q3_UNet; label="UNet", linewidth=2, markersize=14)
        rangebars!(ax, 1:128, q4_UNet, q2_UNet)
        scatterlines!(
            ax,
            q3_ViT;
            label="ViT",
            marker=:rect,
            linewidth=2,
            markersize=14,
        )
        rangebars!(ax, 1:128, q4_ViT, q2_ViT)
        Legend(
            fig[1, 2],
            ax;
            patchsize=size_patch,
            rowgap=gap_row,
            labelsize=size_legend,
        )
        return current_figure()
    end
    wsave(projectdir(dir_save * "time_all.pdf"), fig)
    # Time
    T = size(results_UNet[1], 2)
    map(1:T) do t
        quants_UNet = map(1:L) do l
            return quantile(results_UNet[l][:, t, :])
        end
        q2_UNet = circshift(map(q -> q[2], quants_UNet), 63)
        q3_UNet = circshift(map(q -> q[3], quants_UNet), 63)
        q4_UNet = circshift(map(q -> q[4], quants_UNet), 63)
        quants_ViT = map(1:L) do l
            return quantile(results_ViT[l][:, t, :])
        end
        q2_ViT = circshift(map(q -> q[2], quants_ViT), 63)
        q3_ViT = circshift(map(q -> q[3], quants_ViT), 63)
        q4_ViT = circshift(map(q -> q[4], quants_ViT), 63)
        title = "Equivariance Error (Time: $t)"
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
            ylims!(ax, 0.0, 0.065)
            scatterlines!(ax, q3_UNet; label="UNet", linewidth=2, markersize=14)
            rangebars!(ax, 1:128, q4_UNet, q2_UNet)
            scatterlines!(
                ax,
                q3_ViT;
                label="ViT",
                marker=:rect,
                linewidth=2,
                markersize=14,
            )
            rangebars!(ax, 1:128, q4_ViT, q2_ViT)
            Legend(
                fig[1, 2],
                ax;
                patchsize=size_patch,
                rowgap=gap_row,
                labelsize=size_legend,
            )
            return current_figure()
        end
        wsave(projectdir(dir_save * "time_$t.pdf"), fig)
        return nothing
    end
    return nothing
end
##
function plot_eqv_lines_y()
    ##
    dir_save = "results/HatMatrix/Figures/eqv_err/lines_y/"
    ##
    ket = "ket_C_smse"
    ##
    ket_gs = map(0:127) do dy
        if dy == 0
            return "g_identity"
        else
            return "g_shift_x_0_y_$(dy)"
        end
    end
    ## UNet
    name_model = "UNet"
    results_paths_UNet = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_UNet = map(ket_gs) do ket_g
        results_paths_g = filter(
            p ->
                occursin(ket_g * "_J_" * ket, p) &&
                occursin("errs/mse_scaled/", p),
            results_paths_UNet,
        )
        results_g = map(p -> load(p)[ket_g], results_paths_g)
        return stack(results_g)
    end
    ## ViT
    name_model = "ViT"
    results_paths_ViT = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_ViT = map(ket_gs) do ket_g
        results_paths_g = filter(
            p ->
                occursin(ket_g * "_J_" * ket, p) &&
                occursin("errs/mse_scaled/", p),
            results_paths_ViT,
        )
        results_g = map(p -> load(p)[ket_g], results_paths_g)
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
    L = size(results_ViT, 1)
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    # All
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
    title = "Equivariance Error (All Times)"
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
        ylims!(ax, 0.0, 0.065)
        scatterlines!(ax, q3_UNet; label="UNet", linewidth=2, markersize=14)
        rangebars!(ax, 1:128, q4_UNet, q2_UNet)
        scatterlines!(
            ax,
            q3_ViT;
            label="ViT",
            marker=:rect,
            linewidth=2,
            markersize=14,
        )
        rangebars!(ax, 1:128, q4_ViT, q2_ViT)
        Legend(
            fig[1, 2],
            ax;
            patchsize=size_patch,
            rowgap=gap_row,
            labelsize=size_legend,
        )
        return current_figure()
    end
    wsave(projectdir(dir_save * "time_all.pdf"), fig)
    # Time
    T = size(results_UNet[1], 2)
    map(1:T) do t
        quants_UNet = map(1:L) do l
            return quantile(results_UNet[l][:, t, :])
        end
        q2_UNet = circshift(map(q -> q[2], quants_UNet), 63)
        q3_UNet = circshift(map(q -> q[3], quants_UNet), 63)
        q4_UNet = circshift(map(q -> q[4], quants_UNet), 63)
        quants_ViT = map(1:L) do l
            return quantile(results_ViT[l][:, t, :])
        end
        q2_ViT = circshift(map(q -> q[2], quants_ViT), 63)
        q3_ViT = circshift(map(q -> q[3], quants_ViT), 63)
        q4_ViT = circshift(map(q -> q[4], quants_ViT), 63)
        title = "Equivariance Error (Time: $t)"
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
            ylims!(ax, 0.0, 0.065)
            scatterlines!(ax, q3_UNet; label="UNet", linewidth=2, markersize=14)
            rangebars!(ax, 1:128, q4_UNet, q2_UNet)
            scatterlines!(
                ax,
                q3_ViT;
                label="ViT",
                marker=:rect,
                linewidth=2,
                markersize=14,
            )
            rangebars!(ax, 1:128, q4_ViT, q2_ViT)
            Legend(
                fig[1, 2],
                ax;
                patchsize=size_patch,
                rowgap=gap_row,
                labelsize=size_legend,
            )
            return current_figure()
        end
        wsave(projectdir(dir_save * "time_$t.pdf"), fig)
        return nothing
    end
    return nothing
end
