function plot_diag()
    ##
    name_datas = (:CE, :NS)
    name_models = (:UNet, :ViT)
    ##
    for name_data in name_datas
        if name_data == :CE
            bra_fns = (:bra_C_smse, :bra_C_mass, :bra_C_energy)
        elseif name_data == :NS
            bra_fns = (:bra_C_smse,)
        end
        for name_model in name_models
            for bra_fn in bra_fns
                try
                    fig = plot_diag(name_model, name_data, bra_fn)
                catch e
                end
            end
        end
    end
    ##
    return nothing
end
function plot_diag(name_model::Symbol, name_data::Symbol, bra_fn::Symbol)
    ##
    bra_g = :g_identity
    if bra_fn == :bra_C_smse
        loss_fn = :loss_smse
        label_y = "Response class (SMSE)"
        title = L"$H_{CC}$ Transferability (%$(name_model))"
    elseif bra_fn == :bra_C_mass
        loss_fn = :loss_mass
        label_y = "Response class (Mass)"
        title = L"$H_{MC}$ Transferability (%$(name_model))"
    elseif bra_fn == :bra_C_energy
        loss_fn = :loss_energy
        label_y = "Response class (Energy)"
        title = "Transferability ($(name_model))"
        title = L"$H_{EC}$ Transferability (%$(name_model))"
    end
    ##
    T = 16
    N = 3
    c_inds = vec(collect(CartesianIndices((T, N, T, N))))
    ##
    vals = map(Iterators.product(1:N, 1:N)) do (n1, n2)
        c_inds_nn = filter(ind -> ind[2] == n1 && ind[4] == n2, c_inds)
        hats = get_hat_normed(
            name_model, name_data, bra_fn, bra_g, loss_fn, c_inds_nn
        )
        return mean(hats)
    end
    ##
    padding_figure = (3, 5, 1, 1)
    if name_data == :CE
        ticks_x = ([1, 2, 3], ["RP", "CRP", "RPUI"])
        ticks_y = ([1, 2, 3], ["RP", "CRP", "RPUI"])
    elseif name_data == :NS
        ticks_x = ([1, 2, 3], ["BB", "Gauss", "Sines"])
        ticks_y = ([1, 2, 3], ["BB", "Gauss", "Sines"])
    end
    label_x = "Input class (SMSE)"
    size_title = 20
    size_label = 18
    size_tick_label = 16
    size_figure = (300, 300)
    ##
    range_color = (0, 2.5)
    map_color = :amp
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
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
        )
        hm = heatmap!(ax, vals; colorrange=range_color, colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/$(name_model)/$(bra_fn)/diag.pdf"
    )
    wsave(path_save, fig)
    ## Auto
    map_color = :binary
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
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
        )
        hm = heatmap!(ax, vals; colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/$(name_model)/$(bra_fn)/diag_auto.pdf"
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
