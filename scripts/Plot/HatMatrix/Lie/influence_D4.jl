##
function plot_influence_D4()
    ##
    names_data = (:CE, :NS)
    ##
    for name_data in names_data
        try
            fig = plot_influence_D4(name_data)
        catch e
            println(e)
        end
    end
    ##
    return nothing
end
function plot_influence_D4(
    name_model::Symbol, name_data::Symbol; normalization::Symbol=:standard
)
    ##
    if name_data == :CE
        epoch = 150
    elseif name_data == :NS
        epoch = 100
    end
    ## Measurements
    bra_fn = :loss_mse_scaled
    bra_gs = [
        :g_identity,
        :g_rotate_90,
        :g_rotate_180,
        :g_rotate_270,
        :g_flip,
        :g_flip_rotate_90,
        :g_flip_rotate_180,
        :g_flip_rotate_270,
    ]
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
        return quantile(vec(stack(hats)))[2:4]
    end
    ##
    return hats_g
end
function plot_influence_D4(name_data::Symbol; normalization::Symbol=:standard)
    ##
    hats_UNet = plot_influence_D4(:UNet, name_data; normalization=normalization)
    hats_ViT = plot_influence_D4(:ViT, name_data; normalization=normalization)
    ##
    range_x = collect(1:length(hats_UNet))
    skipper = 0.125
    ranges_x = (range_x .- skipper, range_x .+ skipper)
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    size_figure = (800, 500)
    title = "Gradient Coherence ($(name_data))"
    label_x = "Dihedral Group Element"
    label_y = "Influence"
    size_title = 44
    size_label = 40
    size_tick_label = 40
    width_line = 1
    width_whisker = 16
    size_marker = 20
    size_label_legend = 36
    colgap = 30
    label = [
        L"$e$",
        L"$r$",
        L"$r^2$",
        L"$r^3$",
        L"$s$",
        L"$sr$",
        L"$sr^2$",
        L"$sr^3$",
    ]
    ticks_x = (range_x, label)
    lims_y = (-2.0f0, 6.25f0)
    ##
    U1 = collect(map(e -> e[1], hats_UNet))
    U2 = collect(map(e -> e[2], hats_UNet))
    U3 = collect(map(e -> e[3], hats_UNet))
    V1 = collect(map(e -> e[1], hats_ViT))
    V2 = collect(map(e -> e[2], hats_ViT))
    V3 = collect(map(e -> e[3], hats_ViT))
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            ylabel=label_y,
            ylabelsize=size_label,
            xlabel=label_x,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            xticks=ticks_x,
            xminorticksvisible=false,
        )
        ylims!(ax, lims_y)
        p1 = scatter!(ax, ranges_x[2], V2; markersize=size_marker, label="ViT")
        rangebars!(
            ax,
            ranges_x[2],
            V1,
            V3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        p2 = scatter!(ax, ranges_x[1], U2; markersize=size_marker, label="UNet")
        rangebars!(
            ax,
            ranges_x[1],
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
        "HatMatrix/$(name_data)/Eqv/$(normalization)/influence_D4.pdf"
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
