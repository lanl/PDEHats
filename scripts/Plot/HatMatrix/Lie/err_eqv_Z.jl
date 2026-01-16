##
function plot_err_eqv_Z()
    ##
    names_data = (:CE, :NS)
    names_model = (:UNet, :ViT)
    ##
    for name_data in names_data
        for name_model in names_model
            try
                fig = plot_err_eqv_Z(name_model, name_data, :Horizontal)
            catch e
                println(e)
            end
            try
                fig = plot_err_eqv_Z(name_model, name_data, :Vertical)
            catch e
                println(e)
            end
        end
    end
    ##
    for name_data in names_data
        try
            fig = plot_err_eqv_Z_Q3(name_data, :Horizontal)
        catch e
            println(e)
        end
        try
            fig = plot_err_eqv_Z_Q3(name_data, :Vertical)
        catch e
            println(e)
        end
    end
    ##
    return nothing
end
function plot_err_eqv_Z(
    name_model::Symbol, name_data::Symbol, direction::Symbol
)
    ##
    loss_fn = :loss_smse
    bra_fn = :bra_C_smse
    L = 128
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
    range_l = circshift(collect(0:(L - 1)), div(L, 2))
    if direction == :Horizontal
        bra_gs = map(range_l) do dx
            if dx == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_$(dx)_y_0")
            end
        end
    elseif direction == :Vertical
        bra_gs = map(range_l) do dy
            if dy == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_0_y_$(dy)")
            end
        end
    end
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
        errs_seed = map(1:length(seeds)) do s
            return quantile(vec(stack(errs_array[s, :])))[2:4]
        end
        return errs_seed
    end
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    size_figure = (800, 500)
    title = "Symmetry Loss ($(name_model), $(name_data))"
    label_x = "Translation ($(direction))"
    label_y = "Relative SMSE"
    size_title = 44
    size_label = 40
    size_tick_label = 40
    width_line = 1
    width_whisker = 12
    size_marker = 14
    size_label_legend = 36
    gap_row = 4
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    range_x = collect(1:length(range_l))
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            ylabelsize=size_label,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            xticks=xticks,
        )
        quantiles = ("Q1", "Q2", "Q3")
        for q in 2:3
            quant = quantiles[q]
            vals = collect(map(e -> quantile(stack(e)[q, :]), errs_g))
            vals_1 = map(v -> v[1], vals)
            vals_2 = map(v -> v[2], vals)
            vals_3 = map(v -> v[3], vals)
            scatterlines!(
                ax,
                range_x,
                vals_2;
                markersize=size_marker,
                label="$(quant)",
            )
            rangebars!(
                ax,
                range_x,
                vals_1,
                vals_3;
                whiskerwidth=width_whisker,
                linewidth=width_line,
            )
        end
        axislegend(; position=:rt, labelsize=size_label_legend, rowgap=gap_row)
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "Eqv/$(name_data)/$(name_model)/err_eqv_$(direction).pdf"
    )
    wsave(path_save, fig)
    ##
    return nothing
end
##
function plot_err_eqv_Z_Q3(
    name_model::Symbol, name_data::Symbol, direction::Symbol
)
    ##
    loss_fn = :loss_smse
    bra_fn = :bra_C_smse
    L = 128
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
    range_l = circshift(collect(0:(L - 1)), div(L, 2))
    if direction == :Horizontal
        bra_gs = map(range_l) do dx
            if dx == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_$(dx)_y_0")
            end
        end
    elseif direction == :Vertical
        bra_gs = map(range_l) do dy
            if dy == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_0_y_$(dy)")
            end
        end
    end
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
        errs_seed = map(1:length(seeds)) do s
            return quantile(vec(stack(errs_array[s, :])))[2:4]
        end
        return errs_seed
    end
    return errs_g
end
function plot_err_eqv_Z_Q3(name_data::Symbol, direction::Symbol)
    ##
    errs_g_UNet = plot_err_eqv_Z_Q3(:UNet, name_data, direction)
    errs_g_ViT = plot_err_eqv_Z_Q3(:ViT, name_data, direction)
    ##
    q = 3
    vals_UNet = collect(map(e -> quantile(stack(e)[q, :]), errs_g_UNet))
    UNet_1 = map(v -> v[1], vals_UNet)
    UNet_2 = map(v -> v[2], vals_UNet)
    UNet_3 = map(v -> v[3], vals_UNet)
    vals_ViT = collect(map(e -> quantile(stack(e)[q, :]), errs_g_ViT))
    ViT_1 = map(v -> v[1], vals_ViT)
    ViT_2 = map(v -> v[2], vals_ViT)
    ViT_3 = map(v -> v[3], vals_ViT)
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    size_figure = (800, 500)
    title = "Symmetry Loss (Q3, $(name_data))"
    label_x = "Translation ($(direction))"
    label_y = "Relative SMSE"
    size_title = 44
    size_label = 40
    size_tick_label = 40
    width_line = 1
    width_whisker = 12
    size_marker = 14
    size_label_legend = 36
    gap_row = 4
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    range_x = collect(1:length(errs_g_UNet))
    if name_data == :CE
        lims_y = (0.95, 1.075)
    elseif name_data == :NS
        lims_y = (0.95, 1.12)
    end
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            ylabelsize=size_label,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            xticks=xticks,
        )
        ylims!(ax, lims_y)
        scatterlines!(
            ax,
            range_x,
            ViT_2;
            markersize=size_marker,
            label="ViT",
            marker=:diamond,
        )
        rangebars!(
            ax,
            range_x,
            ViT_1,
            ViT_3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatterlines!(
            ax,
            range_x,
            UNet_2;
            markersize=size_marker,
            label="UNet",
            marker=:circle,
        )
        rangebars!(
            ax,
            range_x,
            UNet_1,
            UNet_3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(; position=:rt, labelsize=size_label_legend, rowgap=gap_row)
        return current_figure()
    end
    ##
    path_save = plotsdir("Eqv/$(name_data)/err_eqv_$(direction)_Q3.pdf")
    wsave(path_save, fig)
    ##
    return nothing
end
