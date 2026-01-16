##
function plot_rkhs()
    ##
    name_datas = (:CE, :NS)
    ##
    for name_data in name_datas
        if name_data == :CE
            bra_fns = (:bra_C_smse, :bra_C_mass, :bra_C_energy)
        elseif name_data == :NS
            bra_fns = (:bra_C_smse,)
        end
        for bra_fn in bra_fns
            try
                fig = plot_rkhs(name_data, bra_fn)
            catch e
                println(e)
            end
        end
    end
    ##
    return nothing
end
##
function plot_rkhs(name_data::Symbol, bra_fn::Symbol)
    ##
    vals_ViT, range_d = plot_rkhs(:ViT, name_data, bra_fn)
    vals_ViT_1 = map(v -> v[1], vals_ViT)
    vals_ViT_2 = map(v -> v[2], vals_ViT)
    vals_ViT_3 = map(v -> v[3], vals_ViT)
    ##
    vals_UNet, _ = plot_rkhs(:UNet, name_data, bra_fn)
    vals_UNet_1 = map(v -> v[1], vals_UNet)
    vals_UNet_2 = map(v -> v[2], vals_UNet)
    vals_UNet_3 = map(v -> v[3], vals_UNet)
    ## Plotting
    padding_figure = (1, 15, 1, 1)
    size_figure = (400, 250)
    size_title = 22
    size_label = 20
    size_label_legend = 20
    size_tick_label = 18
    size_marker = 10
    width_line = 1
    width_whisker = 12
    gap_row = 0.25
    lims_y = (-1, 3)
    ##
    if bra_fn == :bra_C_smse
        label_y = "Influence (SMSE)"
        title = L"$H_{CC}$ Interpolation (%$(name_data))"
    elseif bra_fn == :bra_C_mass
        label_y = "Influence (Mass)"
        title = L"$H_{MC}$ Interpolation (%$(name_data))"
    elseif bra_fn == :bra_C_energy
        label_y = "Influence (Energy)"
        title = L"$H_{EC}$ Interpolation (%$(name_data))"
    end
    label_x = "Input Difference (SMSE)"
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
            xticklabelsize=size_tick_label,
            yticklabelsize=size_tick_label,
            titlesize=size_title,
            xscale=log10,
        )
        ylims!(ax, lims_y)
        scatter!(ax, range_d, vals_UNet_2; markersize=size_marker, label="UNet")
        rangebars!(
            ax,
            range_d,
            vals_UNet_1,
            vals_UNet_3;
            whiskerwidth=width_whisker,
        )
        scatter!(ax, range_d, vals_ViT_2; markersize=size_marker, label="ViT")
        rangebars!(
            ax, range_d, vals_ViT_1, vals_ViT_3; whiskerwidth=width_whisker
        )
        axislegend(; labelsize=size_label_legend, rowgap=gap_row)
        return current_figure()
    end
    ##
    path_save = plotsdir("HatMatrix/$(name_data)/$(bra_fn)/rkhs.pdf")
    wsave(path_save, fig)
    ##
    return fig
end
##
function plot_rkhs(name_model::Symbol, name_data::Symbol, bra_fn::Symbol)
    ##
    bra_g = :g_identity
    if bra_fn == :bra_C_smse
        loss_fn = :loss_smse
    elseif bra_fn == :bra_C_mass
        loss_fn = :loss_mass
    elseif bra_fn == :bra_C_energy
        loss_fn = :loss_energy
    end
    diff_fn = :diff_smse
    seeds = (10, 35, 42)
    if (name_model == :ViT) && (name_data == :NS)
        seeds = (10, 42)
    end
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
    )
    ##
    diffs_array = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        diffs = get_diffs(name_data, seed, idx_NT)
        return diffs[diffs .> 0]
    end
    diff_min = quantile(vec(stack(diffs_array)), 0.005)
    diff_max = quantile(vec(stack(diffs_array)), 0.995)
    ##
    B = 25
    overlap = 5
    bin_edges = exp.(collect(range(log(diff_min), log(diff_max), B)))
    ##
    range_d = map(2:(B - 1)) do b
        edge_L = bin_edges[b]
        edge_R = bin_edges[b + 1]
        return (edge_L + edge_R) / 2
    end
    ##
    curves = map(seeds) do seed
        hats_diffs = map(idx_NTs) do idx_NT
            diffs = get_diffs(name_data, seed, idx_NT)
            hats = get_hat_normed(
                name_model,
                name_data,
                bra_fn,
                bra_g,
                loss_fn,
                seed,
                idx_NT,
            )
            errs = get_err(
                name_model,
                name_data,
                bra_fn,
                bra_g,
                loss_fn,
                seed,
                idx_NT,
            )
            return hats[diffs .> 0], diffs[diffs .> 0]
        end
        hats = vec(stack(map(first, hats_diffs)))
        diffs = vec(stack(map(last, hats_diffs)))
        idx_sort = sortperm(diffs)
        hats_sort = hats[idx_sort]
        diffs_sort = diffs[idx_sort]
        #
        bin_idx_seed =
            map((1 + overlap):(length(bin_edges) - 1 - overlap)) do b
                idx = findall(
                    d ->
                        d >= bin_edges[b - overlap] &&
                            d < bin_edges[b + 1 + overlap],
                    stack(diffs_sort),
                )
                return idx
            end
        filter!(b -> ~isempty(b), bin_idx_seed)
        u = map(bin_idx_seed) do idx
            return Float64(mean(hats_sort[idx]))
        end
        t = map(bin_idx_seed) do idx
            return Float64(mean(diffs_sort[idx]))
        end
        #
        λ = 1e-3
        d = 2
        curve = RegularizationSmooth(
            u,
            t,
            d;
            λ=λ,
            alg=:gcv_svd,
            extrapolation=ExtrapolationType.Linear,
        )
        #
        return curve
    end
    ##
    vals = map(range_d) do d
        ds = collect(map(c -> c(d), curves))
        return quantile(ds)[2:4]
    end
    return (vals, range_d)
end
