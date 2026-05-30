##
function plot_rkhs()
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
                        fig = plot_rkhs(
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
##
function plot_rkhs(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int;
    normalization::Symbol=:nothing,
)
    println("RKHS: $(name_model), $(name_data), $(normalization), $(epoch)")
    ##
    vals, range_d = get_rkhs(name_model, name_data, epoch, normalization)
    vals_1 = map(v -> v[1], vals)
    vals_2 = map(v -> v[2], vals)
    vals_3 = map(v -> v[3], vals)
    ## Plotting
    padding_figure = (1, 15, 1, 1)
    size_figure = (400, 250)
    size_title = 18
    size_supertitle = 22
    size_label = 20
    size_label_legend = 20
    size_tick_label = 18
    size_marker = 10
    width_line = 1
    width_whisker = 12
    gap_row = 0.25
    lims_y = (-0.25, 3.75)
    ##
    label_y = "Cross-Influence"
    supertitle = "Gradient Coherence Over Examples"
    title = "($(name_model), $(name_data), Epoch $(epoch))"
    label_x = "Input Difference (SMSE)"
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            xscale=log10,
        )
        ylims!(ax, lims_y)
        scatter!(ax, range_d, vals_2; markersize=size_marker)
        rangebars!(ax, range_d, vals_1, vals_3; whiskerwidth=width_whisker)
        Label(
            fig[0, 1],
            supertitle;
            fontsize=size_supertitle,
            tellwidth=false,
            tellheight=true,
        )
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/$(name_model)/$(normalization)/rkhs/rkhs_epoch_$(epoch).pdf",
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
function get_rkhs(
    name_model::Symbol, name_data::Symbol, epoch::Int, normalization::Symbol
)
    ##
    bra_fn = :loss_mse_scaled
    bra_g = :g_identity
    diff_fn = :diff_smse
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
    diffs_array = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        diffs = get_diffs(name_data, seed, idx_NT)
        return diffs
    end
    diffs_min_array = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        diffs = get_diffs(name_data, seed, idx_NT)
        return minimum(diffs[diffs .> 0])
    end
    ##
    diff_min = minimum(vec(stack(diffs_min_array)))
    diff_max = maximum(vec(stack(diffs_array)))
    ##
    if name_data == :NS
        N_bins = 15
        overlap = 2
    else
        N_bins = 19
        overlap = 1
    end
    bin_edges =
        exp.(
            collect(
                range(log.(diff_min), log.(diff_max), N_bins + 2 * overlap - 1)
            )
        )
    ##
    bin_ranges = map((1 + overlap):(N_bins - overlap)) do b
        edge_L = bin_edges[b - overlap]
        edge_R = bin_edges[b + overlap]
        return (edge_L, edge_R)
    end
    bin_centers = map((1 + overlap):(N_bins - overlap)) do b
        return bin_edges[b]
    end
    ##
    hats_binned_array = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        diffs = get_diffs(name_data, seed, idx_NT; diff_fn=diff_fn)
        hats = get_hat_normed(
            name_model,
            name_data,
            epoch,
            bra_fn,
            bra_g,
            seed,
            idx_NT;
            normalization=normalization,
        )
        hats_binned = map(bin_ranges) do (edge_L, edge_R)
            idx = findall(
                i -> diffs[i] >= edge_L && diffs[i] <= edge_R,
                1:length(diffs),
            )
            hat_idx = map(i -> hats[i], idx)
            if isempty(hat_idx)
                return missing
            else
                return mean(hat_idx)
            end
        end
        return hats_binned
    end
    # [N_bins, seeds, idx_NT]
    hats_binned_array_stacked = stack(hats_binned_array)
    hats_bins = map(1:size(hats_binned_array_stacked, 1)) do b
        val_low = quantile(skipmissing(hats_binned_array_stacked[b, :, :]), 0.25f0)
        val_mid = quantile(
            skipmissing(hats_binned_array_stacked[b, :, :]), 0.50f0
        )
        val_high = quantile(
            skipmissing(hats_binned_array_stacked[b, :, :]), 0.75f0
        )
        return (val_low, val_mid, val_high)
    end
    ##
    return (hats_bins, bin_centers)
end
