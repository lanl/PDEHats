##
function plot_horizon()
    ##
    kets = ["ket_C_smse", "ket_C_mass"]
    ## UNet
    name_model = "UNet"
    bras = ["bra_C_smse", "bra_C_mass", "bra_C_energy"]
    for ket in kets
        for bra in bras
            if ket == "ket_C_smse" && bra == "bra_C_smse"
                P = 0
                plot_horizon(name_model, bra, ket, P)
            elseif ket == "ket_C_smse" && bra == "bra_C_mass"
                P = 1
                plot_horizon(name_model, bra, ket, P)
            elseif ket == "ket_C_mass" && bra == "bra_C_smse"
                P = 1
                plot_horizon(name_model, bra, ket, P)
            elseif ket == "ket_C_mass" && bra == "bra_C_mass"
                P = 2
                plot_horizon(name_model, bra, ket, P)
            end
        end
    end
    ## ViT
    name_model = "ViT"
    kets = ["ket_C_smse", "ket_C_mass"]
    for ket in kets
        for bra in bras
            if ket == "ket_C_smse" && bra == "bra_C_smse"
                P = 0
                plot_horizon(name_model, bra, ket, P)
            elseif ket == "ket_C_smse" && bra == "bra_C_mass"
                P = 1
                plot_horizon(name_model, bra, ket, P)
            elseif ket == "ket_C_mass" && bra == "bra_C_smse"
                P = 1
                plot_horizon(name_model, bra, ket, P)
            elseif ket == "ket_C_mass" && bra == "bra_C_mass"
                P = 2
                plot_horizon(name_model, bra, ket, P)
            end
        end
    end
    ##
    return nothing
end
##
function plot_horizon(name_model::String, bra::String, ket::String, P::Int)
    ## Load
    results_paths = Archon.find_files_by_suffix(
        projectdir("results/AppHatMatrix/$name_model/"), ".jld2"
    )
    dir_save = "results/AppHatMatrix/Figures/$name_model/"
    ##
    results_paths_bra_ket = filter(
        p -> occursin(bra, p) && occursin(ket, p), results_paths
    )
    ##
    label_y = L"\text{Mean Overlap } H_{%$P}"
    title = "Gradient Alignment ($name_model)"
    ##
    path_save=dir_save * "$(bra)_$(ket)/time_horizon"
    plot_horizon(results_paths_bra_ket, label_y, title, path_save)
    ##
    return nothing
end
##
