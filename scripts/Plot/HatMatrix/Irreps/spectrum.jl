##
function plot_compare(; dir_saved::String=projectdir("results/July2025/"))
    ##
    kets = ["ket_C_smse"] # "ket_C_mass"]
    bras = ["bra_C_smse"] # "bra_C_mass", "bra_C_energy"]
    for ket in kets
        for bra in bras
            plot_compare(bra, ket; dir_saved=dir_saved)
        end
    end
    ##
    return nothing
end
##
function plot_compare(
    bra::String, ket::String; dir_saved::String=projectdir("results/July2025/")
)
    ## Model
    name_model = "UNet"
    ## Load
    results_paths_UNet = PDEHats.find_files_by_suffix(
        projectdir(dir_saved * "HatMatrix/$name_model/"), ".jld2"
    )
    ##
    results_paths_bra_ket_UNet = filter(
        p -> occursin(bra * "_J_g_identity", p) && occursin(ket, p),
        results_paths_UNet,
    )
    ## Model
    name_model = "ViT"
    ## Load
    results_paths_ViT = PDEHats.find_files_by_suffix(
        projectdir(dir_saved * "HatMatrix/$name_model/"), ".jld2"
    )
    results_paths_bra_ket_ViT = filter(
        p -> occursin(bra * "_J_g_identity", p) && occursin(ket, p),
        results_paths_ViT,
    )
    ## Title
    if ket == "ket_C_smse"
        if bra == "bra_C_smse"
            h = 0
        elseif bra == "bra_C_mass"
            h = 1
        elseif bra == "bra_C_energy"
            h = 3
        end
    elseif ket == "ket_C_mass"
        if bra == "bra_C_smse"
            h = 1
        elseif bra == "bra_C_mass"
            h = 2
        elseif bra == "bra_C_energy"
            h = 4
        end
    end
    ##
    path_save = dir_saved * "HatMatrix/Figures/spectra"
    plot_compare(
        results_paths_bra_ket_UNet, results_paths_bra_ket_ViT, path_save; h=h
    )
    ##
    if last(split(bra, "_")) == last(split(ket, "_"))
        path_save = dir_saved * "HatMatrix/Figures/spectra_overlap"
        plot_compare(
            results_paths_bra_ket_UNet,
            results_paths_bra_ket_ViT,
            path_save;
            h=h,
            norm_type=:Overlap,
        )
    end
    ##
    return nothing
end
function plot_compare(
    results_paths_bra_ket_UNet::Vector{<:String},
    results_paths_bra_ket_ViT::Vector{<:String},
    path_save::String;
    norm_type::Symbol=:Frobenius,
    h::Int=0,
)
    ##
    hat_matrices = map(
        p -> load(p)["bra_J_chi_J_ket"], results_paths_bra_ket_UNet
    )
    invariants_array = map(hat_matrices) do M
        if norm_type == :Overlap
            M_overlap = get_M_overlap(M)
            invariants = get_invariants(M_overlap)
        else
            invariants = get_invariants(M)
        end
        return map(norm, eachslice(invariants ./ norm(invariants); dims=1)) .^ 2
    end
    idx = [1, 5, 2, 3, 6, 4]
    invariants_array = map(i -> i[idx], invariants_array)
    invariants_quantiles =
        map(eachslice(stack(invariants_array); dims=1)) do invariants
            q_0, q_1, q_2, q_3, q_4 = quantile(invariants)
            return (q_1, q_2, q_3)
        end
    q_1_array_UNet = map(i -> i[1], invariants_quantiles)
    q_2_array_UNet = map(i -> i[2], invariants_quantiles)
    q_3_array_UNet = map(i -> i[3], invariants_quantiles)
    ## ViT
    hat_matrices = map(
        p -> load(p)["bra_J_chi_J_ket"], results_paths_bra_ket_ViT
    )
    invariants_array = map(hat_matrices) do M
        if norm_type == :Overlap
            M_overlap = get_M_overlap(M)
            invariants = get_invariants(M_overlap)
        else
            invariants = get_invariants(M)
        end
        return map(norm, eachslice(invariants ./ norm(invariants); dims=1)) .^ 2
    end
    invariants_array = map(i -> i[idx], invariants_array)
    invariants_quantiles =
        map(eachslice(stack(invariants_array); dims=1)) do invariants
            q_0, q_1, q_2, q_3, q_4 = quantile(invariants)
            return (q_1, q_2, q_3)
        end
    q_1_array_ViT = map(i -> i[1], invariants_quantiles)
    q_2_array_ViT = map(i -> i[2], invariants_quantiles)
    q_3_array_ViT = map(i -> i[3], invariants_quantiles)
    ##
    title = L"$H_{%$h}$ Spectral Decomposition"
    ticks_x_labels = [
        L"$H_{%$h, [3]}^{(1)}$",
        L"$H_{%$h, [2, 1]}^{(1)}$",
        L"$H_{%$h, [2, 1]}^{(2)}$",
        L"$H_{%$h, [1, 1, 1]}^{(1)}$",
        L"$H_{%$h, [3]}^{(2)}$",
        L"$H_{%$h, [2, 1]}^{(3)}$",
    ]
    ticks_x_labels = ticks_x_labels[idx]
    ticks_x_pos = [1, 2, 3, 4, 5, 6]
    ticks_x = (ticks_x_pos, ticks_x_labels)
    label_x = "Irreducible Representation"
    label_y = "Power Fraction"
    size_label = 16
    size_tick_label = 14
    size_title = 18
    size_marker = 8
    width_whisker = 10
    padding_figure = (1, 5, 1, 1)
    size_fig = (400, 250)
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            titlesize=size_title,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xminorticksvisible=false,
            xticks=ticks_x,
            xticklabelsize=size_tick_label,
        )
        scatter!(ax, q_2_array_UNet; markersize=size_marker, label="UNet")
        rangebars!(
            ax,
            ticks_x_pos,
            q_1_array_UNet,
            q_3_array_UNet;
            whiskerwidth=width_whisker,
        )
        scatter!(ax, q_2_array_ViT; markersize=size_marker, label="ViT")
        rangebars!(
            ax,
            ticks_x_pos,
            q_1_array_ViT,
            q_3_array_ViT;
            whiskerwidth=width_whisker,
        )
        axislegend(; labelsize=size_label, rowgap=1)
        return current_figure()
    end
    ##
    wsave(projectdir(path_save * "_$(h).pdf"), fig)
    ##
    return nothing
end
##
