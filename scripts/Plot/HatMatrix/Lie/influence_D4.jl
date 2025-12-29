##
function plot_influence_D4()
    ##
    name_data = :CE
    dir_save = plotsdir("Eqv/$(name_data)/D4/influence/")
    ## Measurements
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
    loss_fn = :loss_smse
    ## UNet
    hats_UNet = map(bra_gs) do bra_g
        return get_hats(:UNet, name_data, bra_fn, bra_g, loss_fn)
    end
    results_UNet = map(hats_UNet) do hats
        hats_diag = map(hats) do hat
            (T, N) = size(hat)[1:2]
            hat_diag = map(Iterators.product(1:T, 1:N)) do (t, n)
                return hat[t, n, t, n]
            end
        end
        return stack(hats_diag)
    end
    ## ViT
    hats_ViT = map(bra_gs) do bra_g
        return get_hats(:ViT, name_data, bra_fn, bra_g, loss_fn)
    end
    results_ViT = map(hats_ViT) do hats
        hats_diag = map(hats) do hat
            (T, N) = size(hat)[1:2]
            hat_diag = map(Iterators.product(1:T, 1:N)) do (t, n)
                return hat[t, n, t, n]
            end
        end
        return stack(hats_diag)
    end
    ##
    dg = 0.2
    colors = MakiePublication.COLORS[1][1:2]
    gap = 0.65
    size_figure = (800, 450)
    size_title = 40
    size_label = 34
    padding_figure = (1, 5, 1, 1)
    size_tick_label = 32
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
    title = "Influence Function"
    label_x = "Group Element (Dihedral)"
    label_y = "Overlap Value"
    title = "Influence Function"
    ticks_x = (range(1.0f0, 8.0f0, 8), label)
    ## [Time, Batch, Seed, NT]
    N_UNet = prod(size(results_UNet[1]))
    N_ViT = prod(size(results_ViT[1]))
    cats_UNet = vec(stack(
        map(1:8) do g
            cats_g = repeat([g - dg], N_UNet)
            return cats_g
        end,
    ))
    cats_ViT = vec(stack(
        map(1:8) do g
            cats_g = repeat([g + dg], N_ViT)
            return cats_g
        end,
    ))
    ##
    vals_UNet = vec(stack(
        map(1:8) do g
            vals_g = vec(results_UNet[g])
            return vals_g
        end,
    ))
    vals_ViT = vec(stack(
        map(1:8) do g
            vals_g = vec(results_ViT[g])
            return vals_g
        end,
    ))
    ##
    fig =
        with_theme(theme_aps(); colors=colors, figure_padding=padding_figure) do
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
            boxplot!(
                ax,
                cats_UNet,
                vals_UNet;
                label=label,
                gap=gap,
                color=colors[1],
                show_outliers=false,
            )
            boxplot!(
                ax,
                cats_ViT,
                vals_ViT;
                label=label,
                gap=gap,
                color=colors[2],
                show_outliers=false,
            )
            return current_figure()
        end
    # wsave(projectdir(dir_save * "time.pdf"), fig)
    ##
    return fig
end
##
function plot_influence_D4_a()
    ##
    name_data = :CE
    dir_save = plotsdir("Eqv/$(name_data)/D4_a/influence/")
    ## Measurements
    ket_fn = :ket_C_smse
    bra_fn = :bra_C_smse
    ket_g = :g_identity
    bra_gs = (
        :g_rotate_90,
        :g_rotate_180,
        :g_rotate_270,
        :g_flip_rotate_90,
        :g_flip_rotate_180,
        :g_flip_rotate_270,
    )
    loss_fn = :loss_smse
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    ## UNet
    hats_UNet = map(bra_gs) do bra_g
        return get_hats(:UNet, name_data, bra_fn, bra_g, loss_fn)
    end
    results_UNet = map(hats_UNet) do hats
        hats_diag = map(hats) do hat
            (T, N) = size(hat)[1:2]
            hat_diag = map(Iterators.product(1:T, 1:N)) do (t, n)
                return hat[t, n, t, n]
            end
        end
        return stack(hats_diag)
    end
    ## ViT
    hats_ViT = map(bra_gs) do bra_g
        return get_hats(:ViT, name_data, bra_fn, bra_g, loss_fn)
    end
    results_ViT = map(hats_ViT) do hats
        hats_diag = map(hats) do hat
            (T, N) = size(hat)[1:2]
            hat_diag = map(Iterators.product(1:T, 1:N)) do (t, n)
                return hat[t, n, t, n]
            end
        end
        return stack(hats_diag)
    end
    ##
    dg = 0.2
    colors = MakiePublication.COLORS[1][1:2]
    gap = 0.65
    size_figure = (800, 450)
    size_title = 40
    size_label = 34
    padding_figure = (1, 5, 1, 1)
    size_tick_label = 32
    label = [L"$r$", L"$r^2$", L"$r^3$", L"$sr$", L"$sr^2$", L"$sr^3$"]
    title = "Influence Function"
    label_x = "Group Element (Dihedral)"
    label_y = "Overlap Value"
    ticks_x = (range(1.0f0, 6.0f0, 6), label)
    ## [Time, Batch, Seed, NT]
    N_UNet = prod(size(results_UNet[1]))
    N_ViT = prod(size(results_ViT[1]))
    cats_UNet = vec(stack(
        map(1:6) do g
            cats_g = repeat([g - dg], N_UNet)
            return cats_g
        end,
    ))
    cats_ViT = vec(stack(
        map(1:6) do g
            cats_g = repeat([g + dg], N_ViT)
            return cats_g
        end,
    ))
    ##
    vals_UNet = vec(stack(
        map(1:6) do g
            vals_g = vec(results_UNet[g])
            return vals_g
        end,
    ))
    vals_ViT = vec(stack(
        map(1:6) do g
            vals_g = vec(results_ViT[g])
            return vals_g
        end,
    ))
    ##
    fig =
        with_theme(theme_aps(); colors=colors, figure_padding=padding_figure) do
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
            boxplot!(
                ax,
                cats_UNet,
                vals_UNet;
                label=label,
                gap=gap,
                color=colors[1],
                show_outliers=false,
            )
            boxplot!(
                ax,
                cats_ViT,
                vals_ViT;
                label=label,
                gap=gap,
                color=colors[2],
                show_outliers=false,
            )
            return current_figure()
        end
    # wsave(projectdir(dir_save * "time_all.pdf"), fig)
    ##
    return fig
end
function plot_influence_D4_b()
    ##
    name_data = :CE
    dir_save = plotsdir("Eqv/$(name_data)/D4_b/influence/")
    ## Measurements
    ket_fn = :ket_C_smse
    bra_fn = :bra_C_smse
    ket_g = :g_identity
    bra_gs = (:g_identity, :g_flip)
    loss_fn = :loss_smse
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    ## UNet
    hats_UNet = map(bra_gs) do bra_g
        return get_hats(:UNet, name_data, bra_fn, bra_g, loss_fn)
    end
    results_UNet = map(hats_UNet) do hats
        hats_diag = map(hats) do hat
            (T, N) = size(hat)[1:2]
            hat_diag = map(Iterators.product(1:T, 1:N)) do (t, n)
                return hat[t, n, t, n]
            end
        end
        return stack(hats_diag)
    end
    ## ViT
    hats_ViT = map(bra_gs) do bra_g
        return get_hats(:ViT, name_data, bra_fn, bra_g, loss_fn)
    end
    results_ViT = map(hats_ViT) do hats
        hats_diag = map(hats) do hat
            (T, N) = size(hat)[1:2]
            hat_diag = map(Iterators.product(1:T, 1:N)) do (t, n)
                return hat[t, n, t, n]
            end
        end
        return stack(hats_diag)
    end
    ##
    dg = 0.2
    colors = MakiePublication.COLORS[1][1:2]
    gap = 0.65
    size_figure = (800, 450)
    size_title = 40
    size_label = 34
    padding_figure = (1, 5, 1, 1)
    size_tick_label = 32
    label = [L"$e$", L"$s$"]
    title = "Influence Function"
    label_x = "Group Element (Dihedral)"
    label_y = "Overlap Value"
    ticks_x = (range(1.0f0, 2.0f0, 2), label)
    ## [Time, Batch, Seed, NT]
    N_UNet = prod(size(results_UNet[1]))
    N_ViT = prod(size(results_ViT[1]))
    cats_UNet = vec(stack(
        map(1:2) do g
            cats_g = repeat([g - dg], N_UNet)
            return cats_g
        end,
    ))
    cats_ViT = vec(stack(
        map(1:2) do g
            cats_g = repeat([g + dg], N_ViT)
            return cats_g
        end,
    ))
    ##
    vals_UNet = vec(stack(
        map(1:2) do g
            vals_g = vec(results_UNet[g])
            return vals_g
        end,
    ))
    vals_ViT = vec(stack(
        map(1:2) do g
            vals_g = vec(results_ViT[g])
            return vals_g
        end,
    ))
    ##
    fig =
        with_theme(theme_aps(); colors=colors, figure_padding=padding_figure) do
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
            boxplot!(
                ax,
                cats_UNet,
                vals_UNet;
                label=label,
                gap=gap,
                color=colors[1],
                show_outliers=false,
            )
            boxplot!(
                ax,
                cats_ViT,
                vals_ViT;
                label=label,
                gap=gap,
                color=colors[2],
                show_outliers=false,
            )
            return current_figure()
        end
    # wsave(projectdir(dir_save * "time_all.pdf"), fig)
    ##
    return fig
end
