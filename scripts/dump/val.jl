##
using DrWatson
@quickactivate :PDEHats
using CairoMakie, MakiePublication
##
val_paths = PDEHats.find_files(
    projectdir("results/Train/CE/ViT"), "val-fn=loss_mse_scaled", ".jld2"
)
seeds = (10, 35, 42)
##
for seed in seeds
    name_val_fn = "loss_mse_scaled"
    val_paths_seed = filter(p -> occursin("seed_$(seed)", p), val_paths)
    vals_1 = load(
        only(filter(p -> occursin("epoch_1_to_100", p), val_paths_seed))
    )["history_val"]
    @assert length(vals_1) == 100
    path_2 = only(filter(p -> occursin("epoch_101_to_130", p), val_paths_seed))
    dict_2 = load(path_2)
    _vals_2 = load(
        only(filter(p -> occursin("epoch_101_to_130", p), val_paths_seed))
    )["history_val"]
    @assert length(_vals_2) == 130
    vals_2 = _vals_2[101:130]
    dict_2["history_val"] = vals_2
    wsave(path_2, dict_2)
    vals_3 = load(
        only(filter(p -> occursin("epoch_131_to_150", p), val_paths_seed))
    )["history_val"]
    @assert length(vals_3) == 20
    vals = vcat(vals_1, vals_2, vals_3)
end
