function plot_rkhs(name_model::Symbol, name_data::Symbol, bra_fn::Symbol)
    ##
    name_data = :CE
    bra_fns = [:bra_C_smse] #, :bra_C_mass, :bra_C_energy]
    bra_g = :g_identity
    bra_fn = first(bra_fns)
    if bra_fn == :bra_C_smse
        loss_fn = :loss_smse
    end
    name_model = :ViT
    ##
    diff_fn = :diff_smse
    hats, diffs = get_hats_and_diffs(
        name_model, name_data, bra_fn, bra_g, loss_fn; diff_fn=diff_fn
    )
    hats_off_diag = hats[diffs .> 0]
    diffs_off_diag = diffs[diffs .> 0]
    ## Plot
    padding_figure = (1, 1, 1, 1)
    size_figure = (400, 250)
    size_title = 18
    size_label = 16
    size_tick_label = 14
    title = "rkhs"
    label_x = "Distance $(diff_fn)"
    label_y = "Influence"
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
        )
        scatter!(ax, diffs_off_diag, hats_off_diag; markersize=3)
        return current_figure()
    end
    # wsave(projectdir(path_save * ".pdf"), fig)
    ##
    return nothing
end
##
function get_diffs(
    name_data::Symbol,
    seed::Int,
    idx_NT::@NamedTuple{idx_rp::Int, idx_crp::Int, idx_rpui::Int};
    diff_fn::Symbol=:diff_mse,
)
    ##
    dir_diffs = projectdir("results/Diffs/$(name_data)")
    idx_rp = idx_NT.idx_rp
    idx_crp = idx_NT.idx_crp
    idx_rpui = idx_NT.idx_rpui
    name_file_load = savename(
        (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
    )
    dir_load = PDEHats.find_files_by_suffix(dir_diffs, name_file_load * ".jld2")
    path_diff = only(
        filter(
            r -> occursin("seed_$(seed)", r) && occursin("$(diff_fn)", r),
            dir_load,
        ),
    )
    diffs = load(path_diff)["diffs"]
    return diffs
end
function get_hats_and_diffs(
    name_model::Symbol,
    name_data::Symbol,
    bra_fn::Symbol,
    bra_g::Symbol,
    loss_fn::Symbol;
    diff_fn::Symbol=:diff_mse,
)
    ##
    ket_fn = :ket_C_smse
    ket_g = :g_identity
    seeds = (10, 35, 42)
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
    dir_model = projectdir("results/HatMatrix/$(name_data)/$(name_model)")
    name_file_load = "chi_$(ket_g)_J_$(ket_fn)"
    name_file_save = "$(bra_fn)_J_$(bra_g)_" * name_file_load
    result_paths = PDEHats.find_files(dir_model, name_file_save, ".jld2")
    @assert length(result_paths) == N_Obs
    err_paths = PDEHats.find_files_by_suffix(
        projectdir("results/Eqv/$(name_data)/$(loss_fn)/$(name_model)/"),
        "err_eqv_$(bra_g).jld2",
    )
    @assert length(err_paths) == N_Obs
    hats_and_diffs_array =
        map(Iterators.product(seeds, idx_NTs)) do (seed, idx_NT)
            idx_rp = idx_NT.idx_rp
            idx_crp = idx_NT.idx_crp
            idx_rpui = idx_NT.idx_rpui
            dir_batch = savename(
                (idxrp=idx_rp, idxcrp=idx_crp, idxrpui=idx_rpui); equals="_"
            )
            path_hat = only(
                filter(
                    r -> occursin("seed_$(seed)", r) && occursin(dir_batch, r),
                    result_paths,
                ),
            )
            path_err = only(
                filter(
                    r -> occursin("seed_$(seed)", r) && occursin(dir_batch, r),
                    err_paths,
                ),
            )
            hat = load(path_hat)["bra_J_chi_J_ket"]
            errs = load(path_err)["err_eqv"]
            (T, N) = size(errs)
            errs_r = reshape(errs, T, N, 1, 1)
            hat_normed = hat ./ errs_r
            ##
            diffs = get_diffs(name_data, seed, idx_NT; diff_fn=diff_fn)
            ##
            return hat_normed, diffs
        end
    hats, diffs = batch(vec(hats_and_diffs_array))
    return hats, diffs
end
##
