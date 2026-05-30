##
function plot_err_eqv_Z()
    ##
    names_data = (:CE, :NS)
    names_model = (:UNet, :ViT)
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
##
function get_err_eqv_Z(name_model::Symbol, name_data::Symbol, direction::Symbol)
    ##
    L = 128
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
    range_l = circshift(collect(0:(L - 1)), div(L, 2))
    if direction == :Horizontal
        ket_gs = map(range_l) do dx
            if dx == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_$(dx)_y_0")
            end
        end
    elseif direction == :Vertical
        ket_gs = map(range_l) do dy
            if dy == 0
                return Symbol("g_identity")
            else
                return Symbol("g_shift_x_0_y_$(dy)")
            end
        end
    end
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
    return errs_g
end
function plot_err_eqv_Z_Q3(name_data::Symbol, direction::Symbol)
    ##
    errs_g_UNet = get_err_eqv_Z(:UNet, name_data, direction)
    errs_g_ViT = get_err_eqv_Z(:ViT, name_data, direction)
    ##
    Q = 3
    vals_UNet = collect(map(e -> quantile(stack(e)[Q, :])[2:4], errs_g_UNet))
    UNet_1 = map(v -> v[1], vals_UNet)
    UNet_2 = map(v -> v[2], vals_UNet)
    UNet_3 = map(v -> v[3], vals_UNet)
    vals_ViT = collect(map(e -> quantile(stack(e)[Q, :])[2:4], errs_g_ViT))
    ViT_1 = map(v -> v[1], vals_ViT)
    ViT_2 = map(v -> v[2], vals_ViT)
    ViT_3 = map(v -> v[3], vals_ViT)
    ## Plotting
    padding_figure = (1, 20, 1, 1)
    size_figure = (800, 500)
    title = "Symmetry Loss ($(name_data), 3rd Quantile)"
    label_x = "Translation ($(direction))"
    label_y = "Relative Error"
    size_title = 36
    size_label = 36
    size_tick_label = 36
    width_line = 1
    width_whisker = 10
    size_marker = 12
    size_label_legend = 36
    colgap = 30
    xticks = ([4, 34, 64, 94, 124], ["-60", "-30", "0", "30", "60"])
    range_x = collect(1:length(errs_g_UNet))
    lims_y = (0.995, 1.0815)
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
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
        p1 = scatterlines!(
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
        p2 = scatterlines!(
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
    path_save = plotsdir("HatMatrix/$(name_data)/Eqv/err_eqv_$(direction).pdf")
    wsave(path_save, fig)
    ##
    return nothing
end
