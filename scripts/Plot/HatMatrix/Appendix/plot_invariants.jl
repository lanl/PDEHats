function plot_invariants()
    ##
    kets = ["ket_C_smse", "ket_C_mass"]
    ## UNet
    name_model = "UNet"
    bras = ["bra_C_smse", "bra_C_mass", "bra_C_energy"]
    for ket in kets
        for bra in bras
            if ket == "ket_C_smse" && bra == "bra_C_smse"
                P = 0
                plot_invariants(name_model, bra, ket, P)
            elseif ket == "ket_C_smse" && bra == "bra_C_mass"
                P = 1
                plot_invariants(name_model, bra, ket, P)
            elseif ket == "ket_C_mass" && bra == "bra_C_smse"
                P = 1
                plot_invariants(name_model, bra, ket, P)
            elseif ket == "ket_C_mass" && bra == "bra_C_mass"
                P = 2
                plot_invariants(name_model, bra, ket, P)
            end
        end
    end
    ## ViT
    name_model = "ViT"
    for ket in kets
        for bra in bras
            if ket == "ket_C_smse" && bra == "bra_C_smse"
                P = 0
                plot_invariants(name_model, bra, ket, P)
            elseif ket == "ket_C_smse" && bra == "bra_C_mass"
                P = 1
                plot_invariants(name_model, bra, ket, P)
            elseif ket == "ket_C_mass" && bra == "bra_C_smse"
                P = 1
                plot_invariants(name_model, bra, ket, P)
            elseif ket == "ket_C_mass" && bra == "bra_C_mass"
                P = 2
                plot_invariants(name_model, bra, ket, P)
            end
        end
    end
    ##
    return nothing
end
function plot_invariants(name_model::String, bra::String, ket::String, P::Int)
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
    path_save=dir_save * "$(bra)_$(ket)/invariants"
    plot_invariants(results_paths_bra_ket, path_save, P)
    ##
    return nothing
end
##
##
function plot_invariants(
    results_paths_bra_ket::Vector{<:String}, path_save::String, P::Int
)
    ##
    hat_matrices = map(p -> load(p)["bra_J_chi_J_ket"], results_paths_bra_ket)
    invariants_array = map(hat_matrices) do M
        invariants = get_invariants(M)
        return invariants
    end
    invariants_mean = eachslice(mean(invariants_array); dims=1)
    range_color = (5.0f-3, 5.0f4)
    plot_invariants(
        invariants_mean, P; range_color=range_color, path_save=path_save
    )
    ##
    return nothing
end
##
function plot_invariants(
    invariants::AbstractVector{<:AbstractArray{Float32}},
    P::Int;
    path_save::String="dir_save_default/irreps",
    range_color::Tuple=(0, 1),
    n::Int=3,
)
    ##
    clip_low = :white
    titles = [
        L"\text{Global Mean: } | H_{%$P, [%$n]}^{(1)}(\tau, t)|",
        L"\text{Row Bias: } | H_{%$P, [%$(n-1), 1]}^{(2)}(\tau, t)|",
        L"\text{Column Bias: } | H_{%$P, [%$(n-1), 1]}^{(1)}(\tau, t)|",
        L"\text{Directional Skew: } | H_{%$P, [%$(n-2),1,1]}^{(1)}(\tau, t)|",
        L"\text{Diagonal Bias: } | H_{%$P, [%$n]}^{(2)}(\tau, t)|",
        L"\text{Residual Overlap: } | H_{%$P, [%$(n-1), 1]}^{(3)}(\tau, t)|",
    ]
    ##
    for (i, invariant) in enumerate(invariants)
        title = titles[i]
        label_x = L"\text{Perturbation time index } (t)"
        label_y = L"\text{Response time index } (\tau)"
        size_label = 14
        size_title = 18
        size_tick_label = 12
        padding_figure = (1, 5, 1, 1)
        size_fig = (300, 300)
        fig = with_theme(theme_aps(); figure_padding=padding_figure) do
            fig = Figure(; size=size_fig)
            ax = Makie.Axis(
                fig[1, 1];
                title=title,
                titlesize=size_title,
                aspect=DataAspect(),
                xlabel=label_x,
                ylabel=label_y,
                xlabelsize=size_label,
                ylabelsize=size_label,
                xticklabelsize=size_tick_label,
                yticklabelsize=size_tick_label,
            )
            hm = heatmap!(
                ax,
                invariant;
                colorrange=range_color,
                colormap=:binary,
                lowclip=clip_low,
                colorscale=log10,
            )
            cb = Colorbar(fig[1, 2], hm; ticklabelsize=size_tick_label)
            rowsize!(fig.layout, 1, Aspect(1, 1))
            return current_figure()
        end
        wsave(projectdir(path_save * "_$i.pdf"), fig)
    end
    return nothing
end
##
