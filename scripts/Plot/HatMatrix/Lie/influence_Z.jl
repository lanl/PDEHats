##
function plot_influence_Z()
    plot_influence_Z(:Horizontal)
    plot_influence_Z(:Vertical)
    return nothing
end
function plot_influence_Z(direction::Symbol)
    ##
    L = 128
    name_data = :CE
    loss_fn = :loss_smse
    bra_fn = :bra_C_smse
    dir_save = plotsdir("Eqv/$(name_data)/$(loss_fn)/Z/influence/")
    ##
    if direction == :Horizontal
        bra_gs = map(0:(L - 1)) do dx
            if dx == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_$(dx)_y_0")
            end
        end
    elseif direction == :Vertical
        bra_gs = map(0:(L - 1)) do dy
            if dy == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_0_y_$(dy)")
            end
        end
    end
    ## UNet
    hats_UNet = map(bra_gs) do bra_g
        return get_hats(:UNet, name_data, bra_fn, bra_g, loss_fn)
    end
    results_UNet = map(hats_UNet) do hats
        hats_diag = map(hats) do hat
            (T, N) = size(hat)[1:2]
            hat_diag = map(Iterators.product(1:T, 1:N)) do (t, n)
                return hat[t, n, t, n]
            end
        end
        return stack(hats_diag)
    end
    ## ViT
    hats_ViT = map(bra_gs) do bra_g
        return get_hats(:ViT, name_data, bra_fn, bra_g, loss_fn)
    end
    results_ViT = map(hats_ViT) do hats
        hats_diag = map(hats) do hat
            (T, N) = size(hat)[1:2]
            hat_diag = map(Iterators.product(1:T, 1:N)) do (t, n)
                return hat[t, n, t, n]
            end
        end
        return stack(hats_diag)
    end
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    label_x = "Translation ($(direction))"
    label_y = "Influence Function"
    size_figure = (800, 450)
    size_patch = (8, 8)
    gap_row = 4
    size_legend = 32
    size_title = 40
    size_label = 34
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
    title = "Influence Function"
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
            xticks=xticks,
            ylabelsize=size_label,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
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
    # wsave(projectdir(dir_save * "time.pdf"), fig)
    ##
    return fig
end
