function plot_heatmap()
    ##
    name_data = :CE
    bra_fns = [:bra_C_smse] #, :bra_C_mass, :bra_C_energy]
    ##
    for bra_fn in bra_fns
        plot_heatmap(name_data, bra_fn)
    end
    ##
    return nothing
end
function plot_heatmap(name_data::Symbol, bra_fn::Symbol)
    ##
    bra_g = :g_identity
    if bra_fn == :bra_C_smse
        loss_fn = :loss_smse
    end
    ## UNet
    hats_UNet = get_hats(:UNet, name_data, bra_fn, bra_g, loss_fn)
    vals_UNet = mean(
        map(hats_UNet) do hat
            return dropdims(mean(hat; dims=(2, 4)); dims=(2, 4))
        end,
    )
    ## ViT
    hats_ViT = get_hats(:ViT, name_data, bra_fn, bra_g, loss_fn)
    vals_ViT = mean(
        map(hats_ViT) do hat
            return dropdims(mean(hat; dims=(2, 4)); dims=(2, 4))
        end,
    )
    ##
    range_min = min(minimum(vals_UNet), minimum(vals_ViT))
    range_max = max(maximum(vals_UNet), maximum(vals_ViT))
    range_color = (range_min, range_max)
    label_x = L"\text{Perturbation time index } (t)"
    label_y = L"\text{Response time index } (\tau)"
    size_label = 14
    size_title = 18
    size_tick_label = 12
    padding_figure = (1, 5, 1, 1)
    size_fig = (300, 300)
    ## UNet
    title = "Mean Influence (UNet)"
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
        )
        hm = heatmap!(
            ax,
            vals_UNet;
            colorrange=range_color,
            colormap=:binary,
            colorscale=log10,
        )
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    # wsave(plotsdir("heatmap.pdf"), fig)
    ## ViT
    title = "Mean Influence (ViT)"
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
        )
        hm = heatmap!(
            ax,
            vals_ViT;
            colorrange=range_color,
            colormap=:binary,
            colorscale=log10,
        )
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    # wsave(projectdir(path_save * "_$i.pdf"), fig)
    return nothing
end
##
