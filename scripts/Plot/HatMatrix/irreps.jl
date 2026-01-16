##
function plot_irreps()
    ##
    name_data = :CE
    fig = plot_irreps(name_data)
    ##
    name_data = :NS
    fig = plot_irreps(name_data)
    ##
    return nothing
end
function plot_irreps(name_data::Symbol)
    ##
    (q_1s_CE, q_2s_CE, q_3s_CE) = get_irreps(:UNet, name_data)
    (q_1s_NS, q_2s_NS, q_3s_NS) = get_irreps(:ViT, name_data)
    ##
    colors = MakiePublication.COLORS[1][1:3]
    padding_figure = (1, 5, 1, 1)
    size_figure = (800, 450)
    size_title = 40
    size_label = 34
    size_tick_label = 32
    size_marker = 10
    x_range = range(1.0f0, 8.0f0, 8)
    title = "Irreps ($(name_data))"
    label_x = "Sector"
    label_y = "Power"
    width_whisker = 10
    width_line = 1
    size_marker = 10
    size_label_legend = 19
    gap_row = 4
    ##
    fig =
        with_theme(theme_aps(); colors=colors, figure_padding=padding_figure) do
            fig = Figure(; size=size_figure)
            ax = Makie.Axis(
                fig[1, 1];
                title=title,
                titlesize=size_title,
                ylabel=label_y,
                xlabel=label_x,
                ylabelsize=size_label,
                xlabelsize=size_label,
                xticklabelsize=size_tick_label,
                yticklabelsize=size_tick_label,
                xminorticksvisible=false,
            )
            scatter!(ax, q_2s_CE; label="UNet", markersize=size_marker)
            rangebars!(
                ax,
                1:6,
                q_1s_CE,
                q_3s_CE;
                whiskerwidth=width_whisker,
                linewidth=width_line,
            )
            scatter!(ax, q_2s_NS; label="ViT", markersize=size_marker)
            rangebars!(
                ax,
                1:6,
                q_1s_NS,
                q_3s_NS;
                whiskerwidth=width_whisker,
                linewidth=width_line,
            )
            axislegend(;
                labelsize=size_label_legend, position=:rt, rowgap=gap_row
            )
            return current_figure()
        end
    ##
    return fig
end
##
function get_irreps(m::Array{Float32,2})
    ##
    @assert size(m, 1) == size(m, 2)
    n = size(m, 1)
    u = ones(Float32, n) ./ Float32(sqrt(n))
    @tullio P[i, j] := u[i] * u[j]
    Q = I(n) .- P
    delta(i, j) = i == j ? 1 : 0
    #
    normalizer = norm(m)
    m_normalized = m ./ normalizer
    #
    @tullio m_1[i, j] := P[i, a] * m_normalized[a, b] * P[b, j]
    @tullio m_2[i, j] := P[i, a] * m_normalized[a, b] * Q[b, j]
    @tullio m_3[i, j] := Q[i, a] * m_normalized[a, b] * P[b, j]
    #
    @tullio m_4[i, j] :=
        Q[i, a] * (m_normalized[a, b] - m_normalized[b, a]) * Q[b, j] / 2
    @tullio S[i, j] :=
        Q[i, a] * (m_normalized[a, b] + m_normalized[b, a]) * Q[b, j] / 2
    #
    @tullio m_5[i, j] := S[a, a] * Q[i, j] / tr(Q)
    m_6 = S .- m_5
    ## Check norm
    norm_m_1 = norm(m_1)
    norm_m_2 = norm(m_2)
    norm_m_3 = norm(m_3)
    norm_m_4 = norm(m_4)
    norm_m_5 = norm(m_5)
    norm_m_6 = norm(m_6)
    #
    norm_m = sqrt(
        norm_m_1^2 +
        norm_m_2^2 +
        norm_m_3^2 +
        norm_m_4^2 +
        norm_m_5^2 +
        norm_m_6^2,
    )
    @assert isapprox(norm_m, 1)
    ##
    irreps = [m_1, m_2, m_3, m_4, m_5, m_6]
    # Check orthogonality
    map(Iterators.product(1:6, 1:6)) do (i, j)
        overlap = tr(irreps[i]' * irreps[j])
        if i != j
            @assert overlap < eps(Float32)
        end
        return nothing
    end
    irreps = map(i -> i .* normalizer, irreps)
    ##
    return irreps
end
##
function get_irreps(name_model::Symbol, name_data::Symbol)
    ##
    bra_fn = :bra_C_smse
    bra_g = :g_identity
    loss_fn = :loss_smse
    ##
    ket_fn = :ket_C_smse
    ket_g = :g_identity
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
    N_Obs = length(seeds) * length(idx_NTs)
    ##
    irreps_array = map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
        hats = get_hat_normed(
            name_model, name_data, bra_fn, bra_g, loss_fn, seed, idx_NT
        )
        (T, N) = size(hats)[1:2]
        T0 = 3
        T1 = 3
        irreps = map(Iterators.product(T0:T1, T0:T1)) do (t1, t2)
            hat = Float32.(hats[t2, :, t1, :])
            irrep = map(norm, get_irreps(hat))
            return irrep
        end
        return irreps
    end
    ##
    irreps = reshape(stack(stack(irreps_array)), (6, :))
    ##
    qs = map(1:6) do p
        return quantile(irreps[p, :])[2:4]
    end
    q_1s = map(q -> q[1], qs)
    q_2s = map(q -> q[2], qs)
    q_3s = map(q -> q[3], qs)
    return (q_1s, q_2s, q_3s)
end
