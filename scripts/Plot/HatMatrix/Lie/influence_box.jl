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
function plot_influence_box(
    name_model::Symbol, name_data::Symbol; normalization::Symbol=:standard
)
    ##
    if name_data == :CE
        epoch = 150
    elseif name_data == :NS
        epoch = 100
    end
    ##
    L = 17
    bra_fn = :loss_mse_scaled
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
        (; idx_rp=7, idx_crp=7, idx_rpui=7),
        (; idx_rp=8, idx_crp=8, idx_rpui=8),
        (; idx_rp=9, idx_crp=9, idx_rpui=9),
        (; idx_rp=10, idx_crp=10, idx_rpui=10),
        (; idx_rp=11, idx_crp=11, idx_rpui=11),
        (; idx_rp=12, idx_crp=12, idx_rpui=12),
        (; idx_rp=13, idx_crp=13, idx_rpui=13),
        (; idx_rp=14, idx_crp=14, idx_rpui=14),
        (; idx_rp=15, idx_crp=15, idx_rpui=15),
        (; idx_rp=16, idx_crp=16, idx_rpui=16),
        (; idx_rp=17, idx_crp=17, idx_rpui=17),
        (; idx_rp=18, idx_crp=18, idx_rpui=18),
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
        return median(vec(stack(hats)))
    end
    ##
    padding_figure = (15, 15, 1, 5)
    supertitle = "Gradient Coherence Over Translation"
    title = "($(name_model), $(string(name_data)[1:2]))"
    label_x = "Translation (Horizontal)"
    label_y = "Translation (Vertical)"
    xticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    yticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    size_supertitle = 16
    size_title = 16
    size_label = 18
    size_tick_label = 16
    size_fig = (300, 300)
    ## Auto
    map_color = :binary
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
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
        Label(
            fig[0, 1],
            supertitle;
            fontsize=size_supertitle,
            tellwidth=false,
            tellheight=false,
        )
        hm = heatmap!(ax, hats_g; colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/Eqv/$(name_model)/influence_box.pdf"
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
