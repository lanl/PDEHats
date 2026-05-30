##
function plot_err_eqv_D4()
    ##
    names_data = (:CE, :NS)
    ##
    for name_data in names_data
        try
            fig = plot_err_eqv_D4_Q3(name_data)
        catch e
            println(e)
        end
    end
    ##
    return nothing
end
function get_err_eqv_D4(name_model::Symbol, name_data::Symbol)
    ##
    ket_gs = (
        :g_identity,
        :g_rotate_90,
        :g_rotate_180,
        :g_rotate_270,
        :g_flip,
        :g_flip_rotate_90,
        :g_flip_rotate_180,
        :g_flip_rotate_270,
    )
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
    return errs_g
end
function plot_err_eqv_D4_Q3(name_data::Symbol)
    ##
    errs_g_UNet = get_err_eqv_D4(:UNet, name_data)
    errs_g_ViT = get_err_eqv_D4(:ViT, name_data)
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
    ##
    range_x = collect(1:length(errs_g_UNet))
    skipper = 0.125
    ranges_x = (range_x .- skipper, range_x .+ skipper)
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    size_figure = (800, 500)
    title = "Symmetry Loss ($(name_data), Q3)"
    label_x = "Group Element (Dihedral)"
    label_y = "Relative Error"
    size_title = 44
    size_label = 40
    size_tick_label = 40
    width_line = 1
    width_whisker = 16
    size_marker = 16
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
    if name_data == :NS
        scale_y = log10
        position = :rc
        lims_y = (0.50, 10^4)
    elseif name_data == :CE
        scale_y = identity
        position = :rt
        lims_y = (0.925, 1.25)
    end
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            ylabel=label_y,
            xlabel=label_x,
            ylabelsize=size_label,
            xlabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            xticks=ticks_x,
            xminorticksvisible=false,
            yscale=scale_y,
        )
        ylims!(ax, lims_y)
        #
        p1 = scatter!(
            ax, ranges_x[2], ViT_2; markersize=size_marker, label="ViT"
        )
        rangebars!(
            ax,
            ranges_x[2],
            ViT_1,
            ViT_3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        p2 = scatter!(
            ax, ranges_x[1], UNet_2; markersize=size_marker, label="UNet"
        )
        rangebars!(
            ax,
            ranges_x[1],
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
    path_save = plotsdir("HatMatrix/$(name_data)/Eqv/err_eqv_D4.pdf")
    wsave(path_save, fig)
    ##
    return fig
end
##
