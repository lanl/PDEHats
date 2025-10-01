##
function plot_hists()
    ##
    kets = ["ket_C_smse", "ket_C_mass"]
    ## UNet
    name_model = "UNet"
    bras = ["bra_C_smse", "bra_C_mass", "bra_C_energy"]
    for ket in kets
        for bra in bras
            plot_hists(name_model, bra, ket; type_norm=:None)
            if last(split(bra, "_")) == last(split(ket, "_"))
                plot_hists(name_model, bra, ket; type_norm=:Overlap)
            end
        end
    end
    ## ViT
    name_model = "ViT"
    for ket in kets
        for bra in bras
            plot_hists(name_model, bra, ket; type_norm=:None)
            if last(split(bra, "_")) == last(split(ket, "_"))
                plot_hists(name_model, bra, ket; type_norm=:Overlap)
            end
        end
    end
    ##
    return nothing
end
##
function plot_hists(
    name_model::String, bra::String, ket::String; type_norm::Symbol=:None
)
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
    path_save=dir_save * "$(bra)_$(ket)/hist_$(type_norm)"
    ##
    hat_matrices = map(p -> load(p)["bra_J_chi_J_ket"], results_paths_bra_ket)
    irreps_array = map(hat_matrices) do M
        if type_norm == :Overlap
            M_overlap = get_M_overlap(M)
            irreps = get_irreps(M_overlap)
        else
            irreps = get_irreps(M)
        end
        return irreps
    end
    ##
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
    plot_hists(irreps_array; path_save=path_save, h=h)
    ##
    return nothing
end
##
function plot_hists(
    irreps_array::AbstractVector{<:AbstractArray{Float32}};
    path_save::String="dir_save_default/irreps",
    n::Int=3,
    h::Int=0,
)
    ## [3, 3, 6, t, T, B]
    irreps = stack(irreps_array)
    ##
    titles = [
        L"\text{Global Mean: } | H_{%$h, [%$n]}^{(1)}(\tau, t)|",
        L"\text{Row Bias: } | H_{%$h, [%$(n-1), 1]}^{(2)}(\tau, t)|",
        L"\text{Column Bias: } | H_{%$h, [%$(n-1), 1]}^{(1)}(\tau, t)|",
        L"\text{Directional Skew: } | H_{%$h, [%$(n-2),1,1]}^{(1)}(\tau, t)|",
        L"\text{Diagonal Bias: } | H_{%$h, [%$n]}^{(2)}(\tau, t)|",
        L"\text{Residual Overlap: } | H_{%$h, [%$(n-1), 1]}^{(3)}(\tau, t)|",
    ]
    ##
    for i in 1:6
        ##
        vals = vec(selectdim(irreps, 3, i))
        title = titles[i]
        label_x = "Response Value"
        label_y = "Probability Density"
        size_label = 14
        size_title = 18
        padding_figure = (1, 5, 1, 1)
        size_fig = (300, 300)
        #
        lims_x = 1 * std(vals)
        lims_axis = (-lims_x, lims_x)
        fig = PDEHats.hist_1x1_with_legend_b(
            vals,
            lims_axis,
            title,
            label_x,
            label_y;
            y_max=0.5f0,
            size_fig=(300, 300),
            size_title=size_title,
            size_label=size_label,
            N_bins=48,
        )
        ##
        wsave(projectdir(path_save * "_$i.pdf"), fig)
    end
    return nothing
end
##
