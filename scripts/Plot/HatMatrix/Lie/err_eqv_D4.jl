##
function plot_err_eqv_D4()
    ##
    names_data = (:CE, :NS)
    ##
    for name_data in names_data
        try
            fig = plot_err_eqv_D4(name_data)
        catch e
            println(e)
        end
    end
    ##
    return nothing
end
function plot_err_eqv_D4(name_model::Symbol, name_data::Symbol)
    ##
    loss_fn = :loss_smse
    bra_fn = :bra_C_smse
    bra_gs = (
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
        return quantile(vec(stack(errs_array)))[2:4]
    end
    return errs_g
end
function plot_err_eqv_D4(name_data::Symbol)
    ##
    errs_UNet = plot_err_eqv_D4(:UNet, name_data)
    errs_ViT = plot_err_eqv_D4(:ViT, name_data)
    ##
    range_x = collect(1:length(errs_UNet))
    skipper = 0.125
    ranges_x = (range_x .- skipper, range_x .+ skipper)
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    size_figure = (800, 500)
    title = "Symmetry Error ($(name_data))"
    label_x = "Group Element (Dihedral)"
    label_y = "Relative SMSE"
    size_title = 44
    size_label = 40
    size_tick_label = 40
    width_line = 1
    width_whisker = 16
    size_marker = 20
    size_label_legend = 36
    gap_row = 4
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
    U1 = collect(map(e -> e[1], errs_UNet))
    U2 = collect(map(e -> e[2], errs_UNet))
    U3 = collect(map(e -> e[3], errs_UNet))
    V1 = collect(map(e -> e[1], errs_ViT))
    V2 = collect(map(e -> e[2], errs_ViT))
    V3 = collect(map(e -> e[3], errs_ViT))
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
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
        scatter!(ax, ranges_x[2], V2; markersize=size_marker, label="ViT")
        rangebars!(
            ax,
            ranges_x[2],
            V1,
            V3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(ax, ranges_x[1], U2; markersize=size_marker, label="UNet")
        rangebars!(
            ax,
            ranges_x[1],
            U1,
            U3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(;
            position=position, labelsize=size_label_legend, rowgap=gap_row
        )
        return current_figure()
    end
    ##
    path_save = plotsdir("Eqv/$(name_data)/err_eqv_D4.pdf")
    wsave(path_save, fig)
    ##
    return fig
end
##
