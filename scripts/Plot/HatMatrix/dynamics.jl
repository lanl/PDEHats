##
function plot_dynamics()
    ##
    name_datas = (:CE, :NS)
    name_models = (:ViT, :UNet)
    for (name_data, name_model) in Iterators.product(name_datas, name_models)
        plot_dynamics(name_model, name_data)
    end
    ##
    return nothing
end
function plot_dynamics(name_model::Symbol, name_data::Symbol)
    ##
    offs_array, ons_array, epochs = get_dynamics(name_model, name_data)
    ##
    if (name_model == :ViT) && (name_data == :CE)
        @assert epochs[6] == 105
        popat!(epochs, 6)
        popat!(offs_array, 6)
        popat!(ons_array, 6)
    end
    ##
    ons_low = map(v -> v[1], ons_array)
    ons_mid = map(v -> v[2], ons_array)
    ons_high = map(v -> v[3], ons_array)
    #
    offs_low = map(v -> v[1], offs_array)
    offs_mid = map(v -> v[2], offs_array)
    offs_high = map(v -> v[3], offs_array)
    #
    padding_figure = (1, 5, 5, 1)
    size_figure = (400, 250)
    label_x = "Epoch"
    label_y = "Influence (SMSE)"
    title = "Influence Evolution ($(name_model), $(name_data))"
    size_title = 20
    size_label = 20
    size_label_legend = 20
    size_tick_label = 18
    size_marker = 10
    width_line = 1
    width_whisker = 12
    colgap = 30
    #
    if name_data == :CE
        lims_y = (1.0f-4, 4.0f-1)
    elseif name_data == :NS
        lims_y = (6.0f-6, 9)
    end
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            titlesize=size_title,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            yscale=log10,
        )
        ylims!(ax, lims_y)
        p1 = scatter!(
            ax, epochs, ons_mid; markersize=size_marker, label="On-Diagonal"
        )
        rangebars!(
            ax,
            epochs,
            ons_low,
            ons_high;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        p2 = scatter!(
            ax,
            epochs,
            offs_mid;
            markersize=size_marker,
            label="Off-Diagonal",
        )
        rangebars!(
            ax,
            epochs,
            offs_low,
            offs_high;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        Legend(
            fig[2, 1],
            [p1, p2],
            ["On-Diagonal", "Off-Diagonal"];
            orientation=:horizontal,
            colgap=colgap,
            tellwidth=false,
            tellheight=true,
            labelsize=size_label_legend,
        )
        return current_figure()
    end
    ##
    wsave(plotsdir("Dynamics/$(name_data)/$(name_model)/ratio.pdf"), fig)
    return nothing
end
function get_dynamics(name_model::Symbol, name_data::Symbol)
    path_save = projectdir("results/Dynamics")
    name_file_save = "off_on_$(name_data)_$(name_model)"

    dict_dynamics, path_dict_dynamics =
        produce_or_load((;), path_save; filename=name_file_save) do _
            offs_array, ons_array, epochs = _get_dynamics(name_model, name_data)
            dict = Dict(
                "offs_array" => offs_array,
                "ons_array" => ons_array,
                "epochs" => epochs,
            )
            return dict
        end
    offs_array = dict_dynamics["offs_array"]
    ons_array = dict_dynamics["ons_array"]
    epochs = dict_dynamics["epochs"]
    return (offs_array, ons_array, epochs)
end
function _get_dynamics(name_model::Symbol, name_data::Symbol)
    #
    T = 16
    B = 3
    c_inds = CartesianIndices((T, B, T, B))
    c_inds_off = filter(ind -> (ind[1] != ind[3]) || (ind[2] != ind[4]), c_inds)
    c_inds_on = filter(ind -> (ind[1] == ind[3]) && (ind[2] == ind[4]), c_inds)
    #
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
    #
    if name_data == :CE
        if name_model == :ViT
            epochs = [1, 25, 50, 75, 100, 105, 120, 135, 150]
        elseif name_model == :UNet
            epochs = [1, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
        end
    elseif name_data == :NS
        epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    end
    bra_fn = :loss_mse_scaled
    bra_g = :identity
    #
    offs_array = map(epochs) do epoch
        ratio_seed_idx =
            map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
                hat = get_hat(
                    name_model,
                    name_data,
                    epoch,
                    bra_fn,
                    bra_g,
                    seed,
                    idx_NT,
                )
                hat_off = mean(map(c -> hat[c], c_inds_off))
                println("Off: $(hat_off)")
                return hat_off
            end
        return quantile(vec(ratio_seed_idx))[2:4]
    end
    #
    ons_array = map(epochs) do epoch
        ratio_seed_idx =
            map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
                hat = get_hat(
                    name_model,
                    name_data,
                    epoch,
                    bra_fn,
                    bra_g,
                    seed,
                    idx_NT,
                )
                hat_on = mean(map(c -> hat[c], c_inds_on))
                println("On: $(hat_on)")
                return hat_on
            end
        return quantile(vec(ratio_seed_idx))[2:4]
    end
    #
    return (offs_array, ons_array, epochs)
end
