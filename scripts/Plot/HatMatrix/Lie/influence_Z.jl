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
    name_model::Symbol,
    name_data::Symbol,
    direction::Symbol;
    normalization::Symbol=:standard,
)
    ##
    if name_data == :CE
        epoch = 150
    elseif name_data == :NS
        epoch = 100
    end
    ##
    L = 128
    bra_fn = :loss_mse_scaled
    range_L = circshift(collect(0:(L - 1)), div(L, 2))
    if direction == :Horizontal
        bra_gs = map(range_L) do dx
            if dx == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_$(dx)_y_0")
            end
        end
    elseif direction == :Vertical
        bra_gs = map(range_L) do dy
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
    c_inds_diag = filter(ind -> ind[1] == ind[3] && ind[2] == ind[4], c_inds)
    ##
    hats_g = map(bra_gs) do bra_g
        hats = get_hat_normed(
            name_model,
            name_data,
            epoch,
            bra_fn,
            bra_g,
            c_inds_diag;
            normalization=normalization,
        )
        vals = quantile(vec(stack(hats)))[2:4]
        println("$(bra_g): $(vals[2])")
        return vals
    end
    ##
    return hats_g
end
function plot_influence_Z(
    name_data::Symbol, direction::Symbol; normalization::Symbol=:standard
)
    ##
    hats_g_UNet = plot_influence_Z(
        :UNet, name_data, direction; normalization=normalization
    )
    hats_g_ViT = plot_influence_Z(
        :ViT, name_data, direction; normalization=normalization
    )
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    size_figure = (800, 500)
    title = "Gradient Coherence ($(name_data))"
    label_x = "Translation ($(direction))"
    label_y = "Influence"
    size_title = 44
    size_label = 40
    size_tick_label = 40
    width_line = 1
    width_whisker = 6
    size_marker = 8
    size_label_legend = 36
    colgap = 30
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    ##
    if name_data == :NS
        lims_y = (1.20, 5.2)
    elseif name_data == :CE
        lims_y = (0.75, 6.1)
    end
    ##
    range_x = collect(1:length(hats_g_UNet))
    U1 = map(p -> p[1], hats_g_UNet)
    U2 = map(p -> p[2], hats_g_UNet)
    U3 = map(p -> p[3], hats_g_UNet)
    V1 = map(p -> p[1], hats_g_ViT)
    V2 = map(p -> p[2], hats_g_ViT)
    V3 = map(p -> p[3], hats_g_ViT)
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
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
        p1 = scatterlines!(
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
        p2 = scatterlines!(
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
        Legend(
            fig[2, 1],
            [p1, p2],
            ["ViT", "UNet"];
            orientation=:horizontal,
            colgap=colgap,
            tellwidth=false,
            tellheight=true,
            labelsize=size_label_legend,
        )
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/Eqv/$(normalization)/influence_Z_$(direction).pdf",
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
