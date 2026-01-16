function plot_horizon()
    ##
    name_datas = (:CE, :NS)
    name_models = (:UNet, :ViT)
    ##
    for name_data in name_datas
        if name_data == :CE
            bra_fns = (:bra_C_smse, :bra_C_mass, :bra_C_energy)
        elseif name_data == :NS
            bra_fns = (:bra_C_smse,)
        end
        for name_model in name_models
            for bra_fn in bra_fns
                try
                    fig = plot_horizon(name_model, name_data, bra_fn)
                catch e
                    println(e)
                end
            end
        end
    end
    ##
    return nothing
end
##
function plot_horizon(name_model::Symbol, name_data::Symbol, bra_fn::Symbol)
    ##
    bra_g = :g_identity
    if bra_fn == :bra_C_smse
        loss_fn = :loss_smse
        title = L"$H_{CC}$ Extrapolation (%$(name_model), %$(name_data))"
        label_y = "Influence (SMSE)"
    elseif bra_fn == :bra_C_mass
        loss_fn = :loss_mass
        title = L"$H_{MC}$ Extrapolation (%$(name_model), %$(name_data))"
        label_y = "Influence (Mass)"
    elseif bra_fn == :bra_C_energy
        loss_fn = :loss_energy
        title = L"$H_{EC}$ Extrapolation (%$(name_model), %$(name_data))"
        label_y = "Influence (Energy)"
    end
    ##
    T = 16
    N = 3
    c_inds = CartesianIndices((T, N, T, N))
    #
    dT = 8
    range_dT = collect((-dT):dT)
    ## Intra
    vals_intra = map(range_dT) do t
        c_inds_t = filter(ind -> (ind[1] - ind[3] == t) && ind[2] == ind[4], c_inds)
        hats = get_hat_normed(
            name_model, name_data, bra_fn, bra_g, loss_fn, c_inds_t
        )
        return quantile(vec(mean(hats; dims=2)))[2:4]
    end
    intra_1 = map(v -> v[1], vals_intra)
    intra_2 = map(v -> v[2], vals_intra)
    intra_3 = map(v -> v[3], vals_intra)
    ## Inter
    vals_inter = map(range_dT) do t
        c_inds_t = filter(ind -> (ind[1] - ind[3] == t) && ind[2] != ind[4], c_inds)
        hats = get_hat_normed(
            name_model, name_data, bra_fn, bra_g, loss_fn, c_inds_t
        )
        return quantile(vec(mean(hats; dims=2)))[2:4]
    end
    inter_1 = map(v -> v[1], vals_inter)
    inter_2 = map(v -> v[2], vals_inter)
    inter_3 = map(v -> v[3], vals_inter)
    q_1_inter, q_2_inter, q_3_inter = batch(vals_inter)
    ## Plotting
    padding_figure = (1, 5, 1, 1)
    size_figure = (400, 250)
    size_title = 22
    size_label = 20
    size_label_legend = 20
    size_tick_label = 18
    size_marker = 10
    width_line = 1
    width_whisker = 12
    gap_row = 0.25
    lims_y = (-0.25, 7)
    position = :rt
    label_x = "Time Difference"
    ##
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            titlesize=size_title,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
        )
        ylims!(ax, lims_y)
        scatter!(
            ax,
            range_dT,
            intra_2;
            label="Intra-class",
            marker=:circle,
            markersize=size_marker,
        )
        rangebars!(
            ax,
            range_dT,
            intra_1,
            intra_3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(
            ax,
            range_dT,
            inter_2;
            label="Inter-class",
            marker=:diamond,
            markersize=size_marker,
        )
        rangebars!(
            ax,
            range_dT,
            inter_1,
            inter_3;
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(;
            position=position, labelsize=size_label_legend, rowgap=gap_row
        )
        return current_figure()
    end
    ##
    path_save = plotsdir(
        "HatMatrix/$(name_data)/$(name_model)/$(bra_fn)/horizon.pdf"
    )
    wsave(path_save, fig)
    ##
    return fig
end
##
