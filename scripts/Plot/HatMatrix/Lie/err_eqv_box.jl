##
function plot_err_eqv_box()
    ##
    names_data = (:CE, :NS)
    names_model = (:UNet, :ViT)
    ##
    for name_data in names_data
        for name_model in names_model
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
function plot_err_eqv_box_Q3(name_model::Symbol, name_data::Symbol)
    ##
    L = 17
    range_L = collect(0:L)
    ket_gs = map(Iterators.product(range_L, range_L)) do (dx, dy)
        if (dx == 0) && (dy == 0)
            return Symbol("g_identity")
        else
            return Symbol("g_shift_x_$(dx)_y_$(dy)")
        end
    end
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
    errs_g = map(ket_gs) do ket_g
        errs_array =
            map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
                errs_g = get_err(name_model, name_data, seed, idx_NT; ket_g=ket_g)
                errs_e = get_err(name_model, name_data, seed, idx_NT)
                errs = errs_g ./ errs_e
                return errs
            end
        errs_seed = map(1:length(seeds)) do s
            return quantile(vec(stack(errs_array[s, :])))[2:4]
        end
        return errs_seed
    end
    ##
    Q = 3
    vals = collect(map(e -> quantile(stack(e)[Q, :])[2:4], errs_g))
    vals_2 = map(v -> v[2], vals)
    ## Plotting
    padding_figure = (15, 15, 1, 1)
    size_fig = (300, 300)
    supertitle = "Relative Symmetry Loss"
    title = "($(name_model), $(name_data), 3rd Quartile)"
    label_x = "Translation (Horizontal)"
    label_y = "Translation (Vertical)"
    xticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    yticks = ([1, 6, 11, 16], ["0", "5", "10", "15"])
    size_title = 12
    size_supertitle = 18
    size_label = 16
    size_tick_label = 16
    err_abs = maximum(abs.(stack(vals_2))) - 1
    range_color = (1, 1.0815)
    map_color = :amp
    ##
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
            tellheight=true,
        )
        hm = heatmap!(ax, vals_2; colorrange=range_color, colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/Eqv/$(name_model)/err_eqv_box.pdf"
    )
    wsave(path_save, fig)
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
            tellheight=true,
        )
        hm = heatmap!(ax, vals_2; colormap=map_color)
        cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/Eqv/$(name_model)/err_eqv_box_auto.pdf"
    )
    wsave(path_save, fig)
    ##
    return nothing
end
##
