##
function plot_err_eqv_D4()
    ##
    t = 1
    name_data = :CE
    loss_fn = :loss_smse
    dir_save = plotsdir("Eqv/$(name_data)/$(loss_fn)/D4/err_eqv/")
    ket_gs = [
        :g_identity,
        :g_rotate_90,
        :g_rotate_180,
        :g_rotate_270,
        :g_flip,
        :g_flip_rotate_90,
        :g_flip_rotate_180,
        :g_flip_rotate_270,
    ]
    ## UNet
    name_model = :UNet
    results_paths_UNet = PDEHats.find_files(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv",
        ".jld2",
    )
    results_UNet_array = map(ket_gs) do ket_g
        results_paths_g = filter(
            p -> occursin("$(ket_g).jld2", p), results_paths_UNet
        )
        results_g = map(p -> load(p)["err_eqv"], results_paths_g)
        return stack(results_g)
    end
    results_UNet_e = vec(results_UNet_array[1][t, :, :])
    results_UNet_avg = vec(mean(results_UNet_array)[t, :, :])
    results_UNet = [results_UNet_e, results_UNet_avg]
    ## ViT
    name_model = :ViT
    results_paths_ViT = PDEHats.find_files(
        projectdir("results/Eqv/$(name_data)/$(name_loss)/$(name_model)/"),
        "err_eqv",
        ".jld2",
    )
    results_ViT_array = map(ket_gs) do ket_g
        results_paths_g = filter(
            p -> occursin("$(ket_g).jld2", p), results_paths_ViT
        )
        results_g = map(p -> load(p)["err_eqv"], results_paths_g)
        return stack(results_g)
    end
    results_ViT_e = vec(results_ViT_array[1][t, :, :])
    results_ViT_avg = vec(mean(results_ViT_array)[t, :, :])
    results_ViT = [results_ViT_e, results_ViT_avg]
    ## Time
    dg = 0.2
    cats_UNet = vec(
        stack(
            map(1:2) do g
                cats_g = repeat([g - dg], prod(size(results_UNet[g])))
                return cats_g
            end,
        )
    )
    cats_ViT = vec(
        stack(
            map(1:2) do g
                cats_g = repeat([g + dg], prod(size(results_ViT[g])))
                return cats_g
            end,
        )
    )
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
    gap = 0.65
    size_figure = (800, 450)
    size_patch = (8, 8)
    gap_row = 4
    size_title = 40
    size_label = 34
    padding_figure = (1, 5, 1, 1)
    size_tick_label = 32
    label = [L"$e$", L"$D_4$"]
    label = ["No Augmentation", "Symmetry Augmentation"]
    ticks_x = (range(1.0f0, 2.0f0, 2), label)
    title = "Model Equivalence on Deployment"
    label_x = "Group Element (Dihedral)"
    label_y = "Test Error (Log scale)"
    colors = MakiePublication.COLORS[1][1:2]
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
                xlabelsize=size_label,
                xticklabelsize=size_tick_label,
                yticklabelsize=size_tick_label,
                xticks=ticks_x,
                xminorticksvisible=false,
                yticklabelsvisible=false,
            )
            boxplot!(
                ax,
                cats_UNet,
                log.(vals_UNet);
                label=label,
                gap=gap,
                color=colors[1],
                show_outliers=false,
            )
            boxplot!(
                ax,
                cats_ViT,
                log.(vals_ViT);
                label=label,
                gap=gap,
                color=colors[2],
                show_outliers=false,
            )
            return current_figure()
        end
    ##
    path_save = dir_save * "aug_equiv_t$(t).pdf"
    wsave(path_save, fig)
    return nothing
end
