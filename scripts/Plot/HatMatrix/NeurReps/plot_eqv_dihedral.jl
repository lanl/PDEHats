##
function plot_eqv_dihedral()
    ##
    dir_save = "results/HatMatrix/Figures/eqv_err/dihedral/"
    ##
    ket = "ket_C_smse"
    ##
    ket_gs = [
        "g_identity",
        "g_rotate_90",
        "g_rotate_180",
        "g_rotate_270",
        "g_flip",
        "g_flip_rotate_90",
        "g_flip_rotate_180",
        "g_flip_rotate_270",
    ]
    ## UNet
    name_model = "UNet"
    results_paths_UNet = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_UNet = map(ket_gs) do ket_g
        results_paths_g = filter(
            p ->
                occursin(ket_g * "_J_" * ket, p) &&
                occursin("errs/mse_scaled/", p),
            results_paths_UNet,
        )
        results_g = map(p -> load(p)[ket_g], results_paths_g)
        return stack(results_g)
    end
    ## ViT
    name_model = "ViT"
    results_paths_ViT = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_ViT = map(ket_gs) do ket_g
        results_paths_g = filter(
            p ->
                occursin(ket_g * "_J_" * ket, p) &&
                occursin("errs/mse_scaled/", p),
            results_paths_ViT,
        )
        results_g = map(p -> load(p)[ket_g], results_paths_g)
        return stack(results_g)
    end
    ##
    dg = 0.2
    cats_UNet = vec(
        stack(
            map(1:8) do g
                cats_g = repeat([g - dg], prod(size(results_UNet[g])))
                return cats_g
            end,
        )
    )
    cats_ViT = vec(
        stack(
            map(1:8) do g
                cats_g = repeat([g + dg], prod(size(results_ViT[g])))
                return cats_g
            end,
        )
    )
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
    y = 0.0
    # y = -0.25
    points = vec(stack(
        # map(1:8) do g
        map(1:1) do g
            # return ((g - dg / 2, y), (g + dg / 2, y))
            return ((g - 0.25, y), (g + 0.15, y))
        end,
    ))
    text_ann = vec(stack(
        # map(1:8) do g
        map(1:1) do g
            return ("UNet", "ViT")
        end,
    ))
    ##
    gap = 0.65
    size_figure = (800, 450)
    size_title = 40
    size_label = 34
    padding_figure = (1, 5, 1, 1)
    size_patch = (8, 8)
    gap_row = 4
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
    ticks_x = (range(1.0f0, 8.0f0, 8), label)
    title = "Equivariance Error (All Times)"
    label_x = "Group Element (Dihedral)"
    label_y = "SMSE"
    #
    colors = MakiePublication.COLORS[1][1:2]
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
                # show_notch=true,
                label=label,
                gap=gap,
                color=colors[1],
                show_outliers=false,
            )
            boxplot!(
                ax,
                cats_ViT,
                vals_ViT;
                # show_notch=true,
                label=label,
                gap=gap,
                color=colors[2],
                show_outliers=false,
            )
            annotation!(ax, points; text=text_ann)
            return current_figure()
        end
    wsave(projectdir(dir_save * "time_all.pdf"), fig)
    ## Time
    T = size(results_UNet[1], 2)
    map(1:T) do t
        cats_UNet = vec(
            stack(
                map(1:8) do g
                    cats_g = repeat(
                        [g - dg], prod(size(results_UNet[g][:, t, :]))
                    )
                    return cats_g
                end,
            ),
        )
        cats_ViT = vec(
            stack(
                map(1:8) do g
                    cats_g = repeat(
                        [g + dg], prod(size(results_ViT[g][:, t, :]))
                    )
                    return cats_g
                end,
            ),
        )
        vals_UNet = vec(stack(
            map(1:8) do g
                vals_g = vec(results_UNet[g][:, t, :])
                return vals_g
            end,
        ))
        vals_ViT = vec(stack(
            map(1:8) do g
                vals_g = vec(results_ViT[g][:, t, :])
                return vals_g
            end,
        ))
        y = -0.01
        points = vec(stack(
            # map(1:8) do g
            map(1:1) do g
                # return ((g - dg / 2, y), (g + dg / 2, y))
                return ((g - 0.25, y), (g + 0.15, y))
            end,
        ))
        text_ann = vec(stack(
            # map(1:8) do g
            map(1:1) do g
                return ("UNet", "ViT")
            end,
        ))
        #
        gap = 0.65
        size_figure = (800, 450)
        size_patch = (8, 8)
        gap_row = 4
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
        ticks_x = (range(1.0f0, 8.0f0, 8), label)
        title = "Equivariance Error (Time: $t)"
        label_x = "Group Element (Dihedral)"
        label_y = "SMSE"
        #
        colors = MakiePublication.COLORS[1][1:2]
        fig = with_theme(
            theme_aps(); colors=colors, figure_padding=padding_figure
        ) do
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
                # show_notch=true,
                label=label,
                gap=gap,
                color=colors[1],
                show_outliers=false,
            )
            boxplot!(
                ax,
                cats_ViT,
                vals_ViT;
                # show_notch=true,
                label=label,
                gap=gap,
                color=colors[2],
                show_outliers=false,
            )
            annotation!(ax, points; text=text_ann)
            return current_figure()
        end
        wsave(projectdir(dir_save * "time_$t.pdf"), fig)
        return nothing
    end
    ## Field
    F = size(results_UNet[1], 1)
    map(1:F) do f
        cats_UNet = vec(
            stack(
                map(1:8) do g
                    cats_g = repeat(
                        [g - dg], prod(size(results_UNet[g][f, :, :]))
                    )
                    return cats_g
                end,
            ),
        )
        cats_ViT = vec(
            stack(
                map(1:8) do g
                    cats_g = repeat(
                        [g + dg], prod(size(results_ViT[g][f, :, :]))
                    )
                    return cats_g
                end,
            ),
        )
        vals_UNet = vec(stack(
            map(1:8) do g
                vals_g = vec(results_UNet[g][f, :, :])
                return vals_g
            end,
        ))
        vals_ViT = vec(stack(
            map(1:8) do g
                vals_g = vec(results_ViT[g][f, :, :])
                return vals_g
            end,
        ))
        y = 0.0
        points = vec(stack(
            map(1:8) do g
                return ((g - dg / 2, y), (g + dg / 2, y))
            end,
        ))
        text_ann = vec(stack(
            map(1:8) do g
                return ("UNet", "ViT")
            end,
        ))
        #
        gap = 0.65
        size_figure = (800, 450)
        size_title = 30
        size_label = 26
        padding_figure = (1, 5, 1, 1)
        size_patch = (8, 8)
        gap_row = 4
        size_tick_label = 24
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
        ticks_x = (range(1.0f0, 8.0f0, 8), label)
        title = "Equivariance Error (Field: $f)"
        label_x = "Group Element (Dihedral)"
        label_y = "SMSE"
        #
        colors = MakiePublication.COLORS[1][1:2]
        fig = with_theme(
            theme_aps(); colors=colors, figure_padding=padding_figure
        ) do
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
                # show_notch=true,
                label=label,
                gap=gap,
                color=colors[1],
                show_outliers=false,
            )
            boxplot!(
                ax,
                cats_ViT,
                vals_ViT;
                # show_notch=true,
                label=label,
                gap=gap,
                color=colors[2],
                show_outliers=false,
            )
            annotation!(ax, points; text=text_ann)
            return current_figure()
        end
        wsave(projectdir(dir_save * "field_$f.pdf"), fig)
        return nothing
    end
    return nothing
end
