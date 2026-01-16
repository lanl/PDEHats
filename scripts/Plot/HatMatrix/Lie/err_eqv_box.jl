##
function plot_err_eqv_box()
    ##
    names_data = (:CE, :NS)
    names_model = (:UNet, :ViT)
    ##
    for name_data in names_data
        for name_model in names_model
            try
                fig = plot_err_eqv_box(name_model, name_data)
            catch e
                println(e)
            end
            try
                fig = plot_err_eqv_box_Q3(name_model, name_data)
            catch e
                println(e)
            end
        end
    end
    ##
    return nothing
end
##
function plot_err_eqv_box(name_model::Symbol, name_data::Symbol)
    ##
    loss_fn = :loss_smse
    bra_fn = :bra_C_smse
    L = 17
    range_l = collect(0:L)
    bra_gs = map(Iterators.product(range_l, range_l)) do (dx, dy)
        if (dx == 0) && (dy == 0)
            return Symbol("g_identity")
        else
            return Symbol("g_shift_x_$(dx)_y_$(dy)")
        end
    end
    ##
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
    errs_g = map(bra_gs) do bra_g
        errs_array =
            map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
                errs_g = get_err(
                    name_model,
                    name_data,
                    bra_fn,
                    bra_g,
                    loss_fn,
                    seed,
                    idx_NT,
                )
                errs_e = get_err(
                    name_model,
                    name_data,
                    bra_fn,
                    :g_identity,
                    loss_fn,
                    seed,
                    idx_NT,
                )
                errs = errs_g ./ errs_e
                return errs
            end
        return mean(stack(errs_array))
    end
    ## Plotting
    padding_figure = (3, 5, 1, 1)
    size_fig = (300, 300)
    title = "Relative Symmetry Loss ($(name_model), $(name_data))"
    label_x = "Translation (Horizontal)"
    label_y = "Translation (Vertical)"
    xticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    yticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    size_title = 14
    size_label = 18
    size_tick_label = 16
    err_abs = maximum(abs.(stack(errs_g))) - 1
    if name_data == :CE
        range_color = (1, 1.115)
        map_color = :amp
        range_bar = (1, 1 + err_abs)
    elseif name_data == :NS
        range_color = (0.975, 1.025)
        map_color = :balance
        range_bar = (1 - err_abs, 1 + err_abs)
    end
    ##
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
        hm = heatmap!(ax, errs_g; colorrange=range_color, colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir("Eqv/$(name_data)/$(name_model)/err_eqv_box.pdf")
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
        hm = heatmap!(ax, errs_g; colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir("Eqv/$(name_data)/$(name_model)/err_eqv_box_auto.pdf")
    wsave(path_save, fig)
    ##
    return nothing
end
##
function plot_err_eqv_box_Q3(name_model::Symbol, name_data::Symbol)
    ##
    loss_fn = :loss_smse
    bra_fn = :bra_C_smse
    L = 17
    range_l = collect(0:L)
    bra_gs = map(Iterators.product(range_l, range_l)) do (dx, dy)
        if (dx == 0) && (dy == 0)
            return Symbol("g_identity")
        else
            return Symbol("g_shift_x_$(dx)_y_$(dy)")
        end
    end
    ##
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
    errs_g = map(bra_gs) do bra_g
        errs_array =
            map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
                errs_g = get_err(
                    name_model,
                    name_data,
                    bra_fn,
                    bra_g,
                    loss_fn,
                    seed,
                    idx_NT,
                )
                errs_e = get_err(
                    name_model,
                    name_data,
                    bra_fn,
                    :g_identity,
                    loss_fn,
                    seed,
                    idx_NT,
                )
                errs = errs_g ./ errs_e
                return errs
            end
        return quantile(stack(errs_array))[4]
    end
    ## Plotting
    padding_figure = (3, 5, 1, 1)
    size_fig = (300, 300)
    title = "Relative Symmetry Loss (Q3, $(name_model), $(name_data))"
    label_x = "Translation (Horizontal)"
    label_y = "Translation (Vertical)"
    xticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    yticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    size_title = 13
    size_label = 18
    size_tick_label = 16
    err_abs = maximum(abs.(stack(errs_g))) - 1
    if name_data == :CE
        range_color = (1, 1.06)
    elseif name_data == :NS
        range_color = (1, 1.07)
    end
    ##
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
        hm = heatmap!(ax, errs_g; colorrange=range_color, colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir("Eqv/$(name_data)/$(name_model)/err_eqv_box_Q3.pdf")
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
        hm = heatmap!(ax, errs_g; colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "Eqv/$(name_data)/$(name_model)/err_eqv_box_Q3_auto.pdf"
    )
    wsave(path_save, fig)
    ##
    return nothing
end
##
