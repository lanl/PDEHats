##
function plot_influence_Z()
    ##
    name_datas = (:CE, :NS)
    ##
    for name_data in name_datas
        try
            fig = plot_influence_Z(name_data, :Horizontal)
        catch e
            println(e)
        end
        try
            fig = plot_influence_Z(name_data, :Vertical)
        catch e
            println(e)
        end
    end
    ##
    return nothing
end
function plot_influence_Z(
    name_model::Symbol, name_data::Symbol, direction::Symbol
)
    ##
    loss_fn = :loss_smse
    bra_fn = :bra_C_smse
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
    L = 128
    range_l = circshift(collect(0:(L - 1)), div(L, 2))
    if direction == :Horizontal
        bra_gs = map(range_l) do dx
            if dx == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_$(dx)_y_0")
            end
        end
    elseif direction == :Vertical
        bra_gs = map(range_l) do dy
            if dy == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_0_y_$(dy)")
            end
        end
    end
    ##
    T = 16
    N = 3
    c_inds = vec(collect(CartesianIndices((T, N, T, N))))
    ##
    hats_g = map(bra_gs) do bra_g
        c_inds_diag = filter(ind -> ind[1] == ind[3] && ind[2] == ind[4], c_inds)
        hats = get_hat_normed(
            name_model, name_data, bra_fn, bra_g, loss_fn, c_inds_diag
        )
        hats_seeds = map(1:size(hats, 1)) do s
            return mean(stack(hats[s, :]))
        end
        return quantile(hats_seeds)[2:4]
    end
    return hats_g
end
function plot_influence_Z(name_data::Symbol, direction::Symbol)
    ##
    hats_g_UNet = plot_influence_Z(:UNet, name_data, direction)
    hats_g_ViT = plot_influence_Z(:ViT, name_data, direction)
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    size_figure = (800, 500)
    title = "Symmetry Learning ($(name_data))"
    label_x = "Translation ($(direction))"
    label_y = "Influence (SMSE)"
    size_title = 44
    size_label = 40
    size_tick_label = 40
    width_line = 1
    width_whisker = 12
    size_marker = 14
    size_label_legend = 36
    gap_row = 4
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    ##
    if name_data == :NS
        lims_y = (2.6, 4.4)
    elseif name_data == :CE
        lims_y = (2.8, 7.2)
    end
    ##
    skipper = 2
    range_x = collect(1:length(hats_g_UNet))[1:skipper:end]
    U1 = map(p -> p[1], hats_g_UNet)[1:skipper:end]
    U2 = map(p -> p[2], hats_g_UNet)[1:skipper:end]
    U3 = map(p -> p[3], hats_g_UNet)[1:skipper:end]
    V1 = map(p -> p[1], hats_g_ViT)[1:skipper:end]
    V2 = map(p -> p[2], hats_g_ViT)[1:skipper:end]
    V3 = map(p -> p[3], hats_g_ViT)[1:skipper:end]
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
        ylims!(ax, lims_y)
        scatterlines!(
            ax,
            range_x,
            V2;
            markersize=size_marker,
            label="ViT",
            marker=:diamond,
        )
        rangebars!(
            ax,
            range_x,
            V1,
            V3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatterlines!(
            ax,
            range_x,
            U2;
            markersize=size_marker,
            label="UNet",
            marker=:circle,
        )
        rangebars!(
            ax,
            range_x,
            U1,
            U3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(; position=:rt, labelsize=size_label_legend, rowgap=gap_row)
        return current_figure()
    end
    ##
    path_save = plotsdir("Eqv/$(name_data)/influence_Z_$(direction).pdf")
    wsave(path_save, fig)
    ##
    return fig
end
##
