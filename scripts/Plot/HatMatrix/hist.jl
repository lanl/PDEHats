##
function plot_hist()
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
                        fig = plot_hist(
                            name_model, name_data; normalization=normalization
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
function plot_hist(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int;
    normalization::Symbol=:nothing,
)
    println("Hist: $(name_model), $(name_data), $(normalization), $(epoch)")
    ##
    title = "$(normalization) ($(name_model), $(string(name_data)[1:2]), Epoch $(epoch))"
    label_y = "Probability Fraction"
    ##
    vals = get_hat_normed(
        name_model, name_data, epoch; normalization=normalization
    )
    ##
    x_min = quantile(vals, 0.01)
    x_max = quantile(vals, 0.99)
    N_bins = 48
    bins = range(x_min, x_max, N_bins)
    ##
    padding_figure = (1, 15, 1, 1)
    label_x = "Influence"
    size_title = 16
    size_label = 18
    size_tick_label = 16
    size_fig = (400, 400)
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
        )
        xlims!(ax, x_min, x_max)
        hm = hist!(ax, vals; normalization=:probability, bins=bins)
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/$(name_model)/$(normalization)/hist_epoch_$(epoch).pdf",
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
function get_hat_normed(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int;
    normalization::Symbol=:nothing,
)
    ##
    bra_g = :g_identity
    bra_fn = :loss_mse_scaled
    ##
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
    hats_seed_batch = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        hat_normed = get_hat_normed(
            name_model,
            name_data,
            epoch,
            bra_fn,
            bra_g,
            seed,
            idx_NT;
            normalization=normalization,
        )
        return vec(hat_normed)
    end
    ##
    return vec(stack(hats_seed_batch))
end
