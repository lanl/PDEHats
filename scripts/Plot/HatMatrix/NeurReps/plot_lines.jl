##
function plot_lines()
    plot_lines_y()
    plot_lines_x()
    return nothing
end
##
function plot_lines_y()
    ##
    ket = "ket_C_smse"
    bra = "bra_C_smse"
    L = 127
    dx = 0
    ## UNet
    name_model = "UNet"
    results_paths_UNet = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_array_UNet = stack(
        map(0:L) do dy
            if (dx == 0) && (dy == 0)
                bra_g = "g_identity"
            else
                bra_g = "g_shift_x_$(dx)_y_$(dy)_"
            end
            results_paths_bra_ket = filter(
                p -> occursin(bra * "_J_" * bra_g, p) && occursin(ket, p),
                results_paths_UNet,
            )
            results_matrix = map(results_paths_bra_ket) do p
                hat = load(p)["bra_J_chi_J_ket"]
                T = size(hat, 1)
                N = size(hat, 2)
                results = map(Iterators.product(1:T, 1:N)) do (t, n)
                    result = hat[t, n, t, n] ./ norm(hat)
                    return -result
                end
                return stack(results)
            end
            return stack(results_matrix)
        end,
    )
    ## ViT
    name_model = "ViT"
    results_paths_ViT = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_array_ViT = stack(
        map(0:L) do dy
            if (dx == 0) && (dy == 0)
                bra_g = "g_identity"
            else
                bra_g = "g_shift_x_$(dx)_y_$(dy)_"
            end
            results_paths_bra_ket = filter(
                p -> occursin(bra * "_J_" * bra_g, p) && occursin(ket, p),
                results_paths_ViT,
            )
            results_matrix = map(results_paths_bra_ket) do p
                hat = load(p)["bra_J_chi_J_ket"]
                T = size(hat, 1)
                N = size(hat, 2)
                results = map(Iterators.product(1:T, 1:N)) do (t, n)
                    result = hat[t, n, t, n] ./ norm(hat)
                    return -result
                end
                return stack(results)
            end
            return stack(results_matrix)
        end,
    )
    ## Plotting
    dir_save = "results/HatMatrix/Figures/lines_y/"
    padding_figure = (1, 5, 1, 1)
    label_x = "Translation (Vertical)"
    label_y = "Overlap Value"
    size_figure = (800, 450)
    size_patch = (8, 8)
    gap_row = 4
    size_legend = 32
    size_title = 40
    size_label = 34
    size_tick_label = 32
    T = size(results_array_ViT, 1)
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    ## All
    quants_UNet = map(1:(L + 1)) do l
        return quantile(results_array_UNet[:, :, :, l])
    end
    q2_UNet = circshift(map(q -> q[2], quants_UNet), 63)
    q3_UNet = circshift(map(q -> q[3], quants_UNet), 63)
    q4_UNet = circshift(map(q -> q[4], quants_UNet), 63)
    quants_ViT = map(1:(L + 1)) do l
        return quantile(results_array_ViT[:, :, :, l])
    end
    q2_ViT = circshift(map(q -> q[2], quants_ViT), 63)
    q3_ViT = circshift(map(q -> q[3], quants_ViT), 63)
    q4_ViT = circshift(map(q -> q[4], quants_ViT), 63)
    ##
    title = "Gradient Overlap (All Times)"
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
        ylims!(ax, 0.0, 0.11)
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
    ## Time
    map(1:T) do t
        quants_UNet = map(1:(L + 1)) do l
            return quantile(results_array_UNet[t:t, :, :, l])
        end
        q2_UNet = circshift(map(q -> q[2], quants_UNet), 63)
        q3_UNet = circshift(map(q -> q[3], quants_UNet), 63)
        q4_UNet = circshift(map(q -> q[4], quants_UNet), 63)
        quants_ViT = map(1:(L + 1)) do l
            return quantile(results_array_ViT[t:t, :, :, l])
        end
        q2_ViT = circshift(map(q -> q[2], quants_ViT), 63)
        q3_ViT = circshift(map(q -> q[3], quants_ViT), 63)
        q4_ViT = circshift(map(q -> q[4], quants_ViT), 63)
        title = "Gradient Overlap (Time: $t)"
        fig = with_theme(theme_aps(); figure_padding=padding_figure) do
            fig = Figure(; size=size_figure)
            ax = Makie.Axis(
                fig[1, 1];
                title=title,
                xlabel=label_x,
                ylabel=label_y,
                xticks=xticks,
                ylabelsize=size_label,
                xlabelsize=size_label,
                xticklabelsize=size_tick_label,
                yticklabelsize=size_tick_label,
                titlesize=size_title,
            )
            ylims!(ax, 0.0, 0.11)
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
    ##
    return nothing
end
##
function plot_lines_x()
    ##
    ket = "ket_C_smse"
    bra = "bra_C_smse"
    L = 127
    dy = 0
    ## UNet
    name_model = "UNet"
    results_paths_UNet = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_array_UNet = stack(
        map(0:L) do dx
            if (dx == 0) && (dy == 0)
                bra_g = "g_identity"
            else
                bra_g = "g_shift_x_$(dx)_y_$(dy)_"
            end
            results_paths_bra_ket = filter(
                p -> occursin(bra * "_J_" * bra_g, p) && occursin(ket, p),
                results_paths_UNet,
            )
            results_matrix = map(results_paths_bra_ket) do p
                hat = load(p)["bra_J_chi_J_ket"]
                T = size(hat, 1)
                N = size(hat, 2)
                results = map(Iterators.product(1:T, 1:N)) do (t, n)
                    result = hat[t, n, t, n] ./ norm(hat)
                    return -result
                end
                return stack(results)
            end
            return stack(results_matrix)
        end,
    )
    ## ViT
    name_model = "ViT"
    results_paths_ViT = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_array_ViT = stack(
        map(0:L) do dx
            if (dx == 0) && (dy == 0)
                bra_g = "g_identity"
            else
                bra_g = "g_shift_x_$(dx)_y_$(dy)_"
            end
            results_paths_bra_ket = filter(
                p -> occursin(bra * "_J_" * bra_g, p) && occursin(ket, p),
                results_paths_ViT,
            )
            results_matrix = map(results_paths_bra_ket) do p
                hat = load(p)["bra_J_chi_J_ket"]
                T = size(hat, 1)
                N = size(hat, 2)
                results = map(Iterators.product(1:T, 1:N)) do (t, n)
                    result = hat[t, n, t, n] ./ norm(hat)
                    return -result
                end
                return stack(results)
            end
            return stack(results_matrix)
        end,
    )
    ## Plotting
    dir_save = "results/HatMatrix/Figures/lines_x/"
    padding_figure = (1, 1, 1, 1)
    label_x = "Translation (Horizontal)"
    label_y = "Overlap Value"
    size_figure = (800, 450)
    size_patch = (8, 8)
    gap_row = 4
    size_legend = 34
    size_title = 40
    size_label = 34
    size_tick_label = 32
    T = size(results_array_ViT, 1)
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    ## All
    quants_UNet = map(1:(L + 1)) do l
        return quantile(results_array_UNet[:, :, :, l])
    end
    q2_UNet = circshift(map(q -> q[2], quants_UNet), 63)
    q3_UNet = circshift(map(q -> q[3], quants_UNet), 63)
    q4_UNet = circshift(map(q -> q[4], quants_UNet), 63)
    quants_ViT = map(1:(L + 1)) do l
        return quantile(results_array_ViT[:, :, :, l])
    end
    q2_ViT = circshift(map(q -> q[2], quants_ViT), 63)
    q3_ViT = circshift(map(q -> q[3], quants_ViT), 63)
    q4_ViT = circshift(map(q -> q[4], quants_ViT), 63)
    title = "Gradient Overlap (All Times)"
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
        ylims!(ax, 0.0, 0.11)
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
    ## Time
    map(1:T) do t
        quants_UNet = map(1:(L + 1)) do l
            return quantile(results_array_UNet[t:t, :, :, l])
        end
        q2_UNet = circshift(map(q -> q[2], quants_UNet), 63)
        q3_UNet = circshift(map(q -> q[3], quants_UNet), 63)
        q4_UNet = circshift(map(q -> q[4], quants_UNet), 63)
        quants_ViT = map(1:(L + 1)) do l
            return quantile(results_array_ViT[t:t, :, :, l])
        end
        q2_ViT = circshift(map(q -> q[2], quants_ViT), 63)
        q3_ViT = circshift(map(q -> q[3], quants_ViT), 63)
        q4_ViT = circshift(map(q -> q[4], quants_ViT), 63)
        title = "Gradient Overlap (Time: $t)"
        fig = with_theme(theme_aps(); figure_padding=padding_figure) do
            fig = Figure(; size=size_figure)
            ax = Makie.Axis(
                fig[1, 1];
                title=title,
                xlabel=label_x,
                ylabel=label_y,
                xticks=xticks,
                ylabelsize=size_label,
                xlabelsize=size_label,
                xticklabelsize=size_tick_label,
                yticklabelsize=size_tick_label,
                titlesize=size_title,
            )
            ylims!(ax, 0.0, 0.11)
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
    ##
    return nothing
end
##
