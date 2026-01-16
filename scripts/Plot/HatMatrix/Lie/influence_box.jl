##
function plot_influence_box()
    ##
    name_datas = (:CE, :NS)
    name_models = (:UNet, :ViT)
    ##
    for name_data in name_datas
        for name_model in name_models
            try
                fig = plot_influence_box(name_model, name_data)
            catch e
                println(e)
            end
        end
    end
    ##
    return nothing
end
function plot_influence_box(name_model::Symbol, name_data::Symbol)
    ##
    L = 17
    loss_fn = :loss_smse
    bra_fn = :bra_C_smse
    seeds = (10, 35, 42)
    if (name_model == :ViT) && (name_data == :NS)
        seeds = (10, 42)
    end
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    ##
    bra_gs = map(Iterators.product(0:L, 0:L)) do (dx, dy)
        if (dx == 0) && (dy == 0)
            return Symbol("g_identity")
        else
            return Symbol("g_shift_x_$(dx)_y_$(dy)")
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
        return mean(hats)
    end
    ##
    padding_figure = (1, 5, 1, 1)
    title = "Influence ($(name_model), $(string(name_data)[1:2]))"
    label_x = "Translation (Horizontal)"
    label_y = "Translation (Vertical)"
    xticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    yticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    size_title = 18
    size_label = 18
    size_tick_label = 16
    size_fig = (300, 300)
    ##
    if name_data == :CE
        range_color = (0, 6.5)
    elseif name_data == :NS
        range_color = (0, 4.25)
    end
    map_color = :amp
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            aspect=DataAspect(),
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            xticks=xticks,
            yticks=yticks,
        )
        hm = heatmap!(ax, hats_g; colormap=map_color, colorrange=range_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir("Eqv/$(name_data)/$(name_model)/influence_box.pdf")
    wsave(path_save, fig)
    ## Auto
    map_color = :binary
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            aspect=DataAspect(),
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            xticks=xticks,
            yticks=yticks,
        )
        hm = heatmap!(ax, hats_g; colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "Eqv/$(name_data)/$(name_model)/influence_box_auto.pdf"
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
