function plot_diag()
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
                        fig = plot_diag(
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
function plot_diag(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int;
    normalization::Symbol=:nothing,
)
    println("Diag: $(name_model), $(name_data), $(normalization), $(epoch)")
    ##
    bra_g = :g_identity
    bra_fn = :loss_mse_scaled
    supertitle = "Coherence Over Initial Conditions"
    title = "($(name_model), $(string(name_data)[1:2]), Epoch $(epoch))"
    label_y = "Response class"
    label_x = "Input class"
    ##
    T = 16
    N = 3
    c_inds = vec(collect(CartesianIndices((T, N, T, N))))
    ##
    vals = map(Iterators.product(1:N, 1:N)) do (n2, n1)
        c_inds_nn = filter(
            ind -> ind[1] == ind[3] && ind[2] == n2 && ind[4] == n1, c_inds
        )
        hats = get_hat_normed(
            name_model,
            name_data,
            epoch,
            bra_fn,
            bra_g,
            c_inds_nn;
            normalization=normalization,
        )
        val = median(vec(stack(hats)))
        println("Diag: $(val)")
        return val
    end
    ##
    padding_figure = (3, 15, 1, 1)
    if name_data == :CE
        ticks_x = ([1, 2, 3], ["RP", "CRP", "RPUI"])
        ticks_y = ([1, 2, 3], ["RP", "CRP", "RPUI"])
    elseif name_data == :NS
        ticks_x = ([1, 2, 3], ["BB", "Gauss", "Sines"])
        ticks_y = ([1, 2, 3], ["BB", "Gauss", "Sines"])
    end
    size_title = 14
    size_supertitle = 18
    size_label = 18
    size_tick_label = 14
    size_figure = (300, 300)
    ## Auto
    map_color = :binary
    if normalization == :standard
        range_color = (0, 5)
    else
        range_color = (0, maximum(vals))
    end
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            titlesize=size_title,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            xticks=ticks_x,
            yticks=ticks_y,
            aspect=DataAspect(),
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
        "HatMatrix/$(name_data)/$(name_model)/$(normalization)/diag/diag_epoch_$(epoch).pdf",
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
