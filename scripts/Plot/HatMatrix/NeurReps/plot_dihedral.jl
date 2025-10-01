##
function plot_dihedral()
    ##
    dir_save = "results/HatMatrix/Figures/dihedral/"
    ## Measurements
    ket = "ket_C_smse"
    bra = "bra_C_smse"
    bra_gs = map(
        g -> string(g),
        [
            "g_identity",
            "g_rotate_90",
            "g_rotate_180",
            "g_rotate_270",
            "g_flip_chi",
            "g_flip_rotate_90",
            "g_flip_rotate_180",
            "g_flip_rotate_270",
        ],
    )
    # UNet
    name_model = "UNet"
    results_paths_UNet = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_array_UNet = stack(
        map(bra_gs) do bra_g
            results_paths_bra_ket = filter(
                p -> occursin(bra * "_J_" * bra_g, p) && occursin(ket, p),
                results_paths_UNet,
            )
            results_matrix = map(results_paths_bra_ket) do p
                hat = load(p)["bra_J_chi_J_ket"]
                T = size(hat, 1)
                N = size(hat, 2)
                results = map(Iterators.product(1:T, 1:N)) do (t, n)
                    result = hat[t, n, t, n] ./ norm(hat)
                    return -result
                end
                return stack(results)
            end
            return stack(results_matrix)
        end,
    )
    # ViT
    name_model = "ViT"
    results_paths_ViT = Archon.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_array_ViT = stack(
        map(bra_gs) do bra_g
            results_paths_bra_ket = filter(
                p -> occursin(bra * "_J_" * bra_g, p) && occursin(ket, p),
                results_paths_ViT,
            )
            results_matrix = map(results_paths_bra_ket) do p
                hat = load(p)["bra_J_chi_J_ket"]
                T = size(hat, 1)
                N = size(hat, 2)
                results = map(Iterators.product(1:T, 1:N)) do (t, n)
                    result = hat[t, n, t, n] ./ norm(hat)
                    return -result
                end
                return stack(results)
            end
            return stack(results_matrix)
        end,
    )
    ## All
    N_UNet = prod(size(results_array_UNet)[1:3])
    N_ViT = prod(size(results_array_ViT)[1:3])
    dg = 0.2
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
    vals_UNet = vec(stack(
        map(1:8) do g
            vals_g = vec(results_array_UNet[:, :, :, g])
            return vals_g
        end,
    ))
    vals_ViT = vec(stack(
        map(1:8) do g
            vals_g = vec(results_array_ViT[:, :, :, g])
            return vals_g
        end,
    ))
    # y = -0.08
    y = -0.25
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
    title = "Gradient Overlap (All Times)"
    label_x = "Group Element (Dihedral)"
    label_y = "Overlap Value"
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
    T = size(results_array_UNet, 1)
    map(1:T) do t
        title = "Gradient Overlap (Time: $t)"
        N_UNet = prod(size(results_array_UNet[t:t, :, :, :])[1:3])
        N_ViT = prod(size(results_array_ViT[t:t, :, :, :])[1:3])
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
        vals_UNet = vec(
            stack(
                map(1:8) do g
                    vals_g = vec(results_array_UNet[t:t, :, :, g])
                    return vals_g
                end,
            )
        )
        vals_ViT = vec(stack(
            map(1:8) do g
                vals_g = vec(results_array_ViT[t:t, :, :, g])
                return vals_g
            end,
        ))
        #
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
    ##
    return nothing
end
##
