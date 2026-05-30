function plot_horizon()
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
                        fig = plot_horizon(
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
function plot_horizon(
    name_model::Symbol,
    name_data::Symbol,
    epoch::Int;
    normalization::Symbol=:nothing,
)
    ##
    println("Horizon: $(name_model), $(name_data), $(normalization), $(epoch)")
    ##
    bra_g = :g_identity
    bra_fn = :loss_mse_scaled
    supertitle = "Temporal Gradient Coherence"
    title = "($(name_model), $(string(name_data)[1:2]), Epoch $(epoch))"
    label_y = "Response"
    label_x = "Time Difference"
    ##
    T = 16
    N = 3
    c_inds = CartesianIndices((T, N, T, N))
    #
    dT = 11
    range_dT = collect((-dT):dT)
    ## Intra
    vals_intra = map(range_dT) do t
        c_inds_t = filter(ind -> (ind[1] - ind[3] == t) && ind[2] == ind[4], c_inds)
        hats = get_hat_normed(
            name_model,
            name_data,
            epoch,
            bra_fn,
            bra_g,
            c_inds_t;
            normalization=normalization,
        )
        hats_stats = quantile(vec(stack(hats)))[2:4]
        println("Intra: $(hats_stats[2])")
        return hats_stats
    end
    intra_1 = map(v -> v[1], vals_intra)
    intra_2 = map(v -> v[2], vals_intra)
    intra_3 = map(v -> v[3], vals_intra)
    ## Inter
    vals_inter = map(range_dT) do t
        c_inds_t = filter(ind -> (ind[1] - ind[3] == t) && ind[2] != ind[4], c_inds)
        hats = get_hat_normed(
            name_model,
            name_data,
            epoch,
            bra_fn,
            bra_g,
            c_inds_t;
            normalization=normalization,
        )
        hats_stats = quantile(vec(stack(hats)))[2:4]
        println("Inter: $(hats_stats[2])")
        return hats_stats
    end
    inter_1 = map(v -> v[1], vals_inter)
    inter_2 = map(v -> v[2], vals_inter)
    inter_3 = map(v -> v[3], vals_inter)
    ## Plotting
    padding_figure = (1, 15, 1, 1)
    size_figure = (400, 250)
    size_title = 16
    size_supertitle = 20
    size_label = 20
    size_label_legend = 20
    size_tick_label = 18
    size_marker = 12
    width_line = 1
    width_whisker = 12
    gap_row = 0.25
    position = :rt
    ##
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
        )
        if normalization == :standard
            ylims!(ax, (-0.75, 6.25))
        end
        p1 = scatter!(
            ax,
            range_dT,
            intra_2;
            label="Intra-class",
            marker=:circle,
            markersize=size_marker,
        )
        rangebars!(
            ax,
            range_dT,
            intra_1,
            intra_3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        p2 = scatter!(
            ax,
            range_dT,
            inter_2;
            label="Inter-class",
            marker=:utriangle,
            markersize=size_marker,
        )
        rangebars!(
            ax,
            range_dT,
            inter_1,
            inter_3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        Legend(
            fig[2, 1],
            [p1, p2],
            ["Intra-class", "Inter-class"];
            orientation=:horizontal,
            colgap=40,
            tellwidth=false,
            tellheight=true,
            labelsize=size_label_legend,
        )
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
        "HatMatrix/$(name_data)/$(name_model)/$(normalization)/horizon/horizon_epoch_$(epoch).pdf",
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
