function plot_t_series()
    ##
    kets = ["ket_C_smse"] # "ket_C_mass"]
    bras = ["bra_C_smse"] # "bra_C_mass", "bra_C_energy"]
    ## UNet
    name_model = "UNet"
    for ket in kets
        for bra in bras
            plot_t_series(name_model, bra, ket)
        end
    end
    ## ViT
    name_model = "ViT"
    for ket in kets
        for bra in bras
            plot_t_series(name_model, bra, ket)
        end
    end
    ##
    return nothing
end
##
function plot_t_series(name_model::String, bra::String, ket::String)
    ##
    if ket == "ket_C_smse"
        if bra == "bra_C_smse"
            label_y = L"Mean Gradient Overlap $H_0$"
        elseif bra == "bra_C_mass"
            label_y = L"Mean Gradient Overlap $H_1$"
        elseif bra == "bra_C_energy"
            label_y = L"Mean Gradient Overlap $H_3$"
        end
    elseif ket == "ket_C_mass"
        if bra == "bra_C_smse"
            label_y = L"Mean Gradient Overlap $H_1$"
        elseif bra == "bra_C_mass"
            label_y = L"Mean Gradient Overlap $H_2$"
        elseif bra == "bra_C_energy"
            label_y = L"Mean Gradient Overlap $H_4$"
        end
    end
    ## Load
    results_paths = PDEHats.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    dir_save = "results/HatMatrix/Figures/$name_model/"
    ##
    results_paths_bra_ket = filter(
        p -> occursin(bra * "_J_g_identity", p) && occursin(ket, p),
        results_paths,
    )
    ##
    path_save=dir_save * "$(bra)_$(ket)/t_series"
    plot_t_series(results_paths_bra_ket, label_y, name_model, path_save)
    ##
    if last(split(bra, "_")) == last(split(ket, "_"))
        path_save=dir_save * "$(bra)_$(ket)/t_series_overlap"
        plot_t_series(
            results_paths_bra_ket,
            label_y,
            name_model,
            path_save;
            type_norm=:Overlap,
        )
    end
    ##
    return nothing
end
function plot_t_series(
    results_paths_bra_ket::Vector{<:String},
    label_y::AbstractString,
    name_model::String,
    path_save::String;
    type_norm::Symbol=:Frobenius,
)
    hat_matrices = map(p -> load(p)["bra_J_chi_J_ket"], results_paths_bra_ket)
    if type_norm == :Overlap
        hats = map(hat_matrices) do M
            M_overlap = get_M_overlap(M)
            return abs.(M_overlap)
        end
    else
        normalizer = mean(map(norm, hat_matrices))
        hats = map(i -> abs.(i) ./ normalizer, hat_matrices)
    end
    T = size(hats[1], 3)
    map(1:T) do tau
        title = "Alignment with Time $(tau) Data ($name_model)"
        plot_t_series(tau, hats, label_y, title, path_save)
        return nothing
    end
end
##
function plot_t_series(
    tau::Int,
    hats::Vector{<:Array{Float32,4}},
    label_y::AbstractString,
    title::AbstractString,
    path_save::String,
)
    ##
    T = size(hats[1], 1)
    c_inds = CartesianIndices(hats[1])
    # Intra
    vals_intra_summary = map(1:T) do t
        c_inds_t = filter(
            ind -> (ind[3] == tau) && (ind[1] == t) && ind[2] == ind[4],
            c_inds,
        )
        vals_array = map(r -> r[c_inds_t], hats)
        vals = vec(stack(vals_array))
        (_, q_1, q_2, q_3, _) = quantile(vals)
        return (q_1, q_2, q_3)
    end
    q_1_intra, q_2_intra, q_3_intra = batch(vals_intra_summary)
    # Inter
    vals_inter_summary = map(1:T) do t
        c_inds_t = filter(
            ind -> (ind[3] == tau) && (ind[1] == t) && ind[2] != ind[4],
            c_inds,
        )
        vals_array = map(r -> r[c_inds_t], hats)
        vals = vec(stack(vals_array))
        (_, q_1, q_2, q_3, _) = quantile(vals)
        return (q_1, q_2, q_3)
    end
    q_1_inter, q_2_inter, q_3_inter = batch(vals_inter_summary)
    #
    padding_figure = (1, 5, 1, 1)
    label_x = "Time Difference of Response"
    size_title = 18
    size_label = 16
    size_tick_label = 14
    size_figure = (400, 250)
    size_marker = 8
    width_line = 1
    width_whisker = 10
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
            yscale=log10,
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
    wsave(path_save * "_$(tau).pdf", fig)
    ##
    return nothing
end
##
