function plot_time_series()
    ##
    name_data = :CE
    bra_fns = [:bra_C_smse] #, :bra_C_mass, :bra_C_energy]
    name_models = (:UNet, :ViT)
    tau = 1
    ##
    for name_model in name_models
        for bra_fn in bra_fns
            plot_time_series(tau, name_model, name_data, bra_fn)
        end
    end
    ##
    return nothing
end
##
function plot_time_series(
    tau::Int, name_model::Symbol, name_data::Symbol, bra_fn::Symbol
)
    ##
    hats = get_hats(name_model, name_data, bra_fn, bra_g, loss_fn)
    ##
    T = size(hats[1], 1)
    c_inds = CartesianIndices(hats[1])
    ## Intra
    vals_intra = map(1:T) do t
        c_inds_t = filter(
            ind -> (ind[3] == tau) && (ind[1] == t) && ind[2] == ind[4],
            c_inds,
        )
        vals_array = map(hat -> hat[c_inds_t], hats)
        vals = vec(stack(vals_array))
        (_, q_1, q_2, q_3, _) = quantile(vals)
        return (q_1, q_2, q_3)
    end
    q_1_intra, q_2_intra, q_3_intra = batch(vals_intra)
    ## Inter
    vals_inter = map(1:T) do t
        c_inds_t = filter(
            ind -> (ind[3] == tau) && (ind[1] == t) && ind[2] != ind[4],
            c_inds,
        )
        vals_array = map(hat -> hat[c_inds_t], hats)
        vals = vec(stack(vals_array))
        (_, q_1, q_2, q_3, _) = quantile(vals)
        return (q_1, q_2, q_3)
    end
    q_1_inter, q_2_inter, q_3_inter = batch(vals_inter)
    ##
    padding_figure = (1, 5, 1, 1)
    label_x = "Time Difference of Response"
    size_title = 18
    size_label = 16
    size_tick_label = 14
    size_figure = (400, 250)
    size_marker = 8
    width_line = 1
    width_whisker = 10
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
            # yscale=log10,
        )
        t_range_full = (1:T) .- tau
        idx = 1:T
        idx_plot = idx[abs.(t_range_full) .< 11]
        scatter!(
            ax,
            t_range_full[idx_plot],
            q_2_intra[idx_plot];
            label="Intra-class",
        )
        rangebars!(
            ax,
            t_range_full[idx_plot],
            q_1_intra[idx_plot],
            q_3_intra[idx_plot];
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        scatter!(
            ax,
            t_range_full[idx_plot],
            q_2_inter[idx_plot];
            label="Inter-class",
        )
        rangebars!(
            ax,
            t_range_full[idx_plot],
            q_1_inter[idx_plot],
            q_3_inter[idx_plot];
            whiskerwidth=width_whisker,
            linewidth=width_line,
        )
        axislegend(; position=:rt, labelsize=size_label, rowgap=0.25)
        return current_figure()
    end
    #
    # wsave(path_save * "_$(tau).pdf", fig)
    ##
    return fig
end
##
