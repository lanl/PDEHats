function plot_heatmap()
    ##
    name_datas = (:CE, :NS)
    name_models = (:UNet, :ViT)
    normalizations = (
        # :nothing,
        :standard,
        # :cosine
    )
    ##
    for normalization in normalizations
        for name_data in name_datas
            for name_model in name_models
                if name_data == :CE
                    if name_model == :UNet
                        epochs = [1, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
                    elseif name_model == :ViT
                        epochs = [
                            1,
                            25,
                            50,
                            75,
                            100,
                            101,
                            105,
                            120,
                            130,
                            131,
                            135,
                            150,
                        ]
                    end
                elseif name_data == :NS
                    if name_model == :UNet
                        epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                    elseif name_model == :ViT
                        epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                    end
                end
                for epoch in epochs
                    try
                        fig = plot_heatmap(
                            name_model,
                            name_data,
                            epoch;
                            normalization=normalization,
                        )
                    catch e
                        println(e)
                    end
                end
            end
        end
    end
    ##
    return nothing
end
function plot_heatmap(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int;
    normalization::Symbol=:nothing,
)
    println("Heatmap: $(name_model), $(name_data), $(normalization), $(epoch)")
    ##
    bra_g = :g_identity
    bra_fn = :loss_mse_scaled
    supertitle = "Gradient Coherence Over Time"
    title = "($(name_model), $(string(name_data)[1:2]), Epoch $(epoch))"
    label_y = "Response time"
    label_x = "Perturbation time"
    ##
    T = 16
    N = 3
    c_inds = vec(collect(CartesianIndices((T, N, T, N))))
    T_range = 1:T
    ##
    vals = map(Iterators.product(T_range, T_range)) do (t1, t2)
        c_inds_t = filter(
            ind -> ind[1] == t2 && ind[3] == t1 && ind[2] == ind[4], c_inds
        )
        hats = get_hat_normed(
            name_model,
            name_data,
            epoch,
            bra_fn,
            bra_g,
            c_inds_t;
            normalization=normalization,
        )
        val = median(vec(stack(hats)))
        println("Heatmap: $(val), ($(t1), $(t2))")
        return val
    end
    ##
    padding_figure = (1, 15, 1, 10)
    size_title = 16
    size_supertitle = 20
    size_label = 18
    size_tick_label = 16
    size_fig = (300, 300)
    ##
    map_color = :binary
    if normalization == :standard
        range_color = (0, 6.5)
    else
        range_color = (0, maximum(vals))
    end
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
        )
        Label(
            fig[0, 1],
            supertitle;
            fontsize=size_supertitle,
            tellwidth=false,
            tellheight=false,
        )
        hm = heatmap!(ax, vals; colormap=map_color, colorrange=range_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/$(name_model)/$(normalization)/heatmap/heatmap_epoch_$(epoch).pdf",
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
