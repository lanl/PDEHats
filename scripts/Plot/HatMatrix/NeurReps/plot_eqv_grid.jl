##
function plot_eqv_grid()
    ##
    ket = "ket_C_smse"
    ##
    L = 16
    ket_gs = map(Iterators.product((-L):L, (-L):L)) do (dx, dy)
        if dx == 0 && dy == 0
            return "g_identity"
        else
            return "g_shift_x_$(dx)_y_$(dy)"
        end
    end
    ## UNet
    name_model = "UNet"
    dir_save_UNet = "results/HatMatrix/Figures/eqv_err/grid/$name_model/"
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
        return mean(results_g)
    end
    ## ViT
    name_model = "ViT"
    dir_save_ViT = "results/HatMatrix/Figures/eqv_err/grid/$name_model/"
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
        return mean(results_g)
    end
    ## All
    results_array_UNet = stack(
        map(r -> dropdims(mean(r; dims=1); dims=1), results_UNet)
    )
    results_array_ViT = stack(
        map(r -> dropdims(mean(r; dims=1); dims=1), results_ViT)
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
    ## Field
    results_array_UNet = stack(
        map(r -> dropdims(mean(r; dims=2); dims=2), results_UNet)
    )
    results_array_ViT = stack(
        map(r -> dropdims(mean(r; dims=2); dims=2), results_ViT)
    )
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
    ## Field
    F = size(results_array_ViT, 1)
    map(1:F) do f
        vals_UNet = results_array_UNet[f, :, :]
        vals_ViT = results_array_ViT[f, :, :]
        # UNet
        title = "Gradient Overlap (UNet, Field: $f)"
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
        wsave(projectdir(dir_save_UNet * "field_$f.pdf"), fig)
        # ViT
        title = "Gradient Overlap (ViT, Field: $f)"
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
        wsave(projectdir(dir_save_ViT * "field_$f.pdf"), fig)
        return nothing
    end
    ##
    return nothing
end
