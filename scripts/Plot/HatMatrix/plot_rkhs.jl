function plot_rkhs()
    ##
    name_model = "UNet"
    kets = ["ket_C_smse", "ket_C_mass"]
    bras = ["bra_C_smse", "bra_C_mass", "bra_C_energy"]
    ## UNet
    name_model = "UNet"
    for ket in kets
        for bra in bras
            plot_rkhs(name_model, bra, ket)
        end
    end
    ## ViT
    name_model = "ViT"
    for ket in kets
        for bra in bras
            plot_rkhs(name_model, bra, ket)
        end
    end
    ##
    return nothing
end
##
function plot_rkhs(name_model::String, bra::String, ket::String)
    ## Load
    results_paths = PDEHats.find_files_by_suffix(
        projectdir("results/HatMatrix/$name_model/"), ".jld2"
    )
    results_paths_bra_ket = filter(
        p -> occursin(bra * "_J_g_identity", p) && occursin(ket, p),
        results_paths,
    )
    dir_save = "results/HatMatrix/Figures/$name_model/"
    ##
    label_x = L"Input Feature Squared Distance $||x - y||^2_{L_2}$"
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
    label_y = L"Response $H_{%$h}$"
    title = L"$H_{%$h}(x, y)\text{ vs. }||x - y||^2_{L_2} \text{ (%$(name_model))}$"
    ##
    path_save=dir_save * "$(bra)_$(ket)/rkhs_normed"
    plot_rkhs(
        results_paths_bra_ket, label_x, label_y, title, path_save; use_norm=true
    )
    ##
    path_save=dir_save * "$(bra)_$(ket)/rkhs"
    plot_rkhs(
        results_paths_bra_ket,
        label_x,
        label_y,
        title,
        path_save;
        use_norm=false,
    )
    ##
    return nothing
end
function plot_rkhs(
    results_paths_bra_ket::Vector{<:String},
    label_x::AbstractString,
    label_y::AbstractString,
    title::AbstractString,
    path_save::String;
    use_norm::Bool=true,
)
    ##
    overlaps_diffs_array = map(results_paths_bra_ket) do p
        f_0 = join(split(p, "/")[1:(end - 2)], "/")
        f_1 = split(p, "/")[end]
        responses = load(p)["bra_J_chi_J_ket"];
        if use_norm
            normalizer = norm(responses)
            overlap = - responses ./ normalizer
        else
            overlap = - responses
        end
        diffs = load(f_0 * "/diffs/L2/" * f_1)["diffs"];
        return (overlap, diffs)
    end
    overlaps_array, diffs_array = batch(overlaps_diffs_array)
    ##
    diffs_off_diag = diffs_array[diffs_array .> 0]
    overlaps_off_diag = overlaps_array[diffs_array .> 0]
    ##
    plot_rkhs(
        diffs_off_diag, overlaps_off_diag, label_x, label_y, title, path_save
    )
    ##
    return nothing
end
##
function plot_rkhs(
    diffs::Vector{Float32},
    overlaps::Vector{Float32},
    label_x::AbstractString,
    label_y::AbstractString,
    title::AbstractString,
    path_save::AbstractString,
)
    ## Plot
    padding_figure = (1, 1, 1, 1)
    size_figure = (400, 250)
    size_title = 18
    size_label = 16
    size_tick_label = 14
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure(; size=size_figure)
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            xlabelsize=size_label,
            ylabelsize=size_label,
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
        )
        scatter!(ax, diffs, overlaps; markersize=3)
        return current_figure()
    end
    wsave(projectdir(path_save * ".pdf"), fig)
    ##
    return nothing
end
##
