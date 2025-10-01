##
function plot_grid()
    ##
    L = 16
    ket = "ket_C_smse"
    bra = "bra_C_smse"
    ## UNet
    name_model = "UNet"
    results_paths_UNet = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    dir_save_UNet = "results/HatMatrix/Figures/grid/$name_model/"
    results_array_UNet = stack(
        map(Iterators.product((-L):L, (-L):L)) do (dx, dy)
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
                return dropdims(mean(results; dims=2); dims=2)
            end
            return mean(results_matrix)
        end,
    )
    ## ViT
    name_model = "ViT"
    results_paths_ViT = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    dir_save_ViT = "results/HatMatrix/Figures/grid/$name_model/"
    results_array_ViT = stack(
        map(Iterators.product((-L):L, (-L):L)) do (dx, dy)
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
                return dropdims(mean(results; dims=2); dims=2)
            end
            return mean(results_matrix)
        end,
    )
    ## Range Color
    color_min = min(minimum(results_array_UNet), minimum(results_array_ViT))
    color_max = max(maximum(results_array_UNet), maximum(results_array_ViT))
    range_color = (color_min, color_max)
    ## All
    vals_ViT = dropdims(mean(results_array_ViT; dims=1); dims=1)
    vals_UNet = dropdims(mean(results_array_UNet; dims=1); dims=1)
    label_x = "Horizontal Translation"
    label_y = "Vertical Translation"
    ticks = (
        [2, 7, 12, 17, 22, 27, 32], ["-15", "-10", "-5", "0", "5", "10", "15"]
    )
    size_label = 14
    size_title = 16
    size_tick_label = 12
    padding_figure = (1, 5, 1, 1)
    size_fig = (400, 400)
    # UNet
    title = "Gradient Overlap (UNet, All Times)"
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            aspect=DataAspect(),
            xticks=ticks,
            yticks=ticks,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
        )
        hm = heatmap!(ax, vals_UNet; colorrange=range_color, colormap=:binary)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    wsave(projectdir(dir_save_UNet * "time_all.pdf"), fig)
    # ViT
    title = "Gradient Overlap (ViT, All Times)"
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            aspect=DataAspect(),
            xticks=ticks,
            yticks=ticks,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
        )
        hm = heatmap!(ax, vals_ViT; colorrange=range_color, colormap=:binary)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    wsave(projectdir(dir_save_ViT * "time_all.pdf"), fig)
    ## Time
    T = size(results_array_ViT, 1)
    map(1:T) do t
        vals_UNet = results_array_UNet[t, :, :]
        vals_ViT = results_array_ViT[t, :, :]
        # UNet
        title = "Gradient Overlap (UNet, Time: $t)"
        fig = with_theme(theme_aps(); figure_padding=padding_figure) do
            fig = Figure(; size=size_fig)
            ax = Makie.Axis(
                fig[1, 1];
                title=title,
                titlesize=size_title,
                aspect=DataAspect(),
                xticks=ticks,
                yticks=ticks,
                xlabel=label_x,
                ylabel=label_y,
                xlabelsize=size_label,
                ylabelsize=size_label,
                xticklabelsize=size_tick_label,
                yticklabelsize=size_tick_label,
            )
            hm = heatmap!(
                ax, vals_UNet; colorrange=range_color, colormap=:binary
            )
            cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
            rowsize!(fig.layout, 1, Aspect(1, 1))
            return current_figure()
        end
        wsave(projectdir(dir_save_UNet * "time_$t.pdf"), fig)
        # ViT
        title = "Gradient Overlap (ViT, Time: $t)"
        fig = with_theme(theme_aps(); figure_padding=padding_figure) do
            fig = Figure(; size=size_fig)
            ax = Makie.Axis(
                fig[1, 1];
                title=title,
                titlesize=size_title,
                aspect=DataAspect(),
                xticks=ticks,
                yticks=ticks,
                xlabel=label_x,
                ylabel=label_y,
                xlabelsize=size_label,
                ylabelsize=size_label,
                xticklabelsize=size_tick_label,
                yticklabelsize=size_tick_label,
            )
            hm = heatmap!(
                ax, vals_ViT; colorrange=range_color, colormap=:binary
            )
            cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
            rowsize!(fig.layout, 1, Aspect(1, 1))
            return current_figure()
        end
        wsave(projectdir(dir_save_ViT * "time_$t.pdf"), fig)
        return nothing
    end
    ##
    return nothing
end
##
