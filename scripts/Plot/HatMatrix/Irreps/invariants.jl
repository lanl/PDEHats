function plot_invariants(; dir_saved::String=projectdir("results/July2025/"))
    ##
    kets = ["ket_C_smse", "ket_C_mass"]
    bras = ["bra_C_smse", "bra_C_mass", "bra_C_energy"]
    ## UNet
    name_model = "UNet"
    for ket in kets
        for bra in bras
            plot_invariants(name_model, bra, ket; dir_saved=dir_saved)
        end
    end
    ## ViT
    name_model = "ViT"
    for ket in kets
        for bra in bras
            plot_invariants(name_model, bra, ket; dir_saved=dir_saved)
        end
    end
    ##
    return nothing
end
function plot_invariants(
    name_model::String,
    bra::String,
    ket::String;
    cutoff::Float32=5.0f-3,
    dir_saved::String=projectdir("results/July2025/"),
)
    ## Load
    results_paths = PDEHats.find_files_by_suffix(
        projectdir(dir_saved * "HatMatrix/$name_model/"), ".jld2"
    )
    dir_save = dir_saved * "HatMatrix/Figures/$name_model/"
    ##
    results_paths_bra_ket = filter(
        p ->
            occursin(bra * "_J_g_identity", p) &&
            occursin("_g_identity_J_" * ket, p),
        results_paths,
    )
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
    path_save = dir_save * "$(bra)_$(ket)/invariants_normed"
    plot_invariants(
        results_paths_bra_ket, path_save, cutoff; type_norm=:Frobenius, h=h
    )
    ##
    path_save = dir_save * "$(bra)_$(ket)/invariants"
    plot_invariants(
        results_paths_bra_ket, path_save, cutoff; type_norm=:None, h=h
    )
    ##
    if last(split(bra, "_")) == last(split(ket, "_"))
        path_save = dir_save * "$(bra)_$(ket)/invariants_overlap"
        plot_invariants(
            results_paths_bra_ket, path_save, cutoff; type_norm=:Overlap, h=h
        )
    end
    ##
    return nothing
end
##
function plot_invariants(
    results_paths_bra_ket::Vector{<:String},
    path_save::String,
    cutoff::AbstractFloat;
    type_norm::Symbol=:None,
    h::Int=0,
)
    ##
    hat_matrices = map(p -> load(p)["bra_J_chi_J_ket"], results_paths_bra_ket)
    ##
    invariants_array = map(hat_matrices) do M
        if type_norm == :Frobenius
            invariants = get_invariants(M)
            return invariants ./ norm(invariants)
        elseif type_norm == :Overlap
            M_overlap = get_M_overlap(M)
            invariants = get_invariants(M_overlap)
            return invariants
        else
            invariants = get_invariants(M)
            return invariants
        end
    end
    invariants_mean = eachslice(mean(invariants_array); dims=1)
    range_color_max = maximum(stack(invariants_mean))
    range_color_min = cutoff * sqrt(mean(abs2.(stack(invariants_mean))))
    range_color = (range_color_min, range_color_max)
    plot_invariants(
        invariants_mean; range_color=range_color, path_save=path_save, h=h
    )
    ##
    return nothing
end
##
function plot_invariants(
    invariants::AbstractVector{<:AbstractArray{Float32}};
    path_save::String="dir_save_default/irreps",
    range_color::Tuple=(0, 1),
    n::Int=3,
    h::Int=0,
)
    ##
    clip_low = :white
    titles = [
        L"\text{Global Mean: } | H_{%$h, [%$n]}^{(1)}(\tau, t)|",
        L"\text{Row Bias: } | H_{%$h, [%$(n-1), 1]}^{(2)}(\tau, t)|",
        L"\text{Column Bias: } | H_{%$h, [%$(n-1), 1]}^{(1)}(\tau, t)|",
        L"\text{Directional Skew: } | H_{%$h, [%$(n-2),1,1]}^{(1)}(\tau, t)|",
        L"\text{Diagonal Bias: } | H_{%$h, [%$n]}^{(2)}(\tau, t)|",
        L"\text{Residual Overlap: } | H_{%$h, [%$(n-1), 1]}^{(3)}(\tau, t)|",
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
