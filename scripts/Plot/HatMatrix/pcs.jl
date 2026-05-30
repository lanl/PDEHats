##
using DrWatson
@quickactivate :PDEHats
#
using Lux, LuxCUDA
using ComponentArrays
using MLUtils
using Random
using Statistics
using CairoMakie, MakiePublication
##
include(projectdir("scripts/HatMatrix/obs_batch.jl"))
include(projectdir("scripts/HatMatrix/err_eqv.jl"))
include(projectdir("scripts/Plot/HatMatrix/HatMatrix.jl"))
##
function plot_pcs()
    ##
    name_models = (:ViT, :UNet)
    name_datas = (:CE, :NS)
    pcs_array = map(
        Iterators.product(name_models, name_datas)
    ) do (name_model, name_data)
        dir_load = projectdir("results/RolloutPCs/$(name_data)/$(name_model)")
        pcs = load(only(readdir(dir_load; join=true)))["pcs"]
        return pcs
    end
    pcs_array_round = map(pcs -> round.(pcs; sigdigits=2), pcs_array)
    ##
    return nothing
end
function get_pcs_t()
    ##
    name_models = (:ViT, :UNet)
    name_datas = (:CE, :NS)
    dT = 16
    T_max = 17
    ##
    pcs_array_dT = map(
        Iterators.product(name_models, name_datas)
    ) do (name_model, name_data)
        pcs = get_pcs_t(name_model, name_data; dT=dT, T_max=T_max)
        dict_pcs = Dict("pcs" => pcs)
        wsave(
            projectdir(
                "results/RolloutPCs/$(name_data)/$(name_model)/pcs_dT.jld2"
            ),
            dict_pcs,
        )
        return pcs
    end
    ##
    return nothing
end
##
function get_pcs_t(
    name_model::Symbol,
    name_data::Symbol;
    dT::Int=5,
    T_max::Int=17,
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    ##
    if name_data == :CE
        epoch = 150
    elseif name_data == :NS
        epoch = 100
    end
    seeds = (10, 35, 42)
    idx_NTs = (
        (; idx_rp=1, idx_crp=1, idx_rpui=1),
        (; idx_rp=2, idx_crp=2, idx_rpui=2),
        (; idx_rp=3, idx_crp=3, idx_rpui=3),
        (; idx_rp=4, idx_crp=4, idx_rpui=4),
        (; idx_rp=5, idx_crp=5, idx_rpui=5),
        (; idx_rp=6, idx_crp=6, idx_rpui=6),
        (; idx_rp=7, idx_crp=7, idx_rpui=7),
        (; idx_rp=8, idx_crp=8, idx_rpui=8),
        (; idx_rp=9, idx_crp=9, idx_rpui=9),
        (; idx_rp=10, idx_crp=10, idx_rpui=10),
        (; idx_rp=11, idx_crp=11, idx_rpui=11),
        (; idx_rp=12, idx_crp=12, idx_rpui=12),
        (; idx_rp=13, idx_crp=13, idx_rpui=13),
        (; idx_rp=14, idx_crp=14, idx_rpui=14),
        (; idx_rp=15, idx_crp=15, idx_rpui=15),
        (; idx_rp=16, idx_crp=16, idx_rpui=16),
        (; idx_rp=17, idx_crp=17, idx_rpui=17),
        (; idx_rp=18, idx_crp=18, idx_rpui=18),
    )
    ##
    ratio_train = 0.65f0
    ratio_val = 0.05f0
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    ##
    T = 16
    N = 3
    # bra_g = :g_identity
    bra_g = :identity
    bra_fn = :loss_mse_scaled
    c_inds = CartesianIndices((T, T_max))
    ## Rollout Error
    errs_array =
        map(Iterators.product(1:dT, idx_NTs, seeds)) do (dt, idx_NT, seed)
            println("Rollout Error: $(dt)")
            println("Rollout Error: $(idx_NT)")
            println("Rollout Error: $(seed)")
            c_inds_dt = map(Tuple, filter(c -> c[2] - c[1] == dt, c_inds))
            # CKPT
            dir_load = projectdir(
                "results/Train/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)/",
            )
            keys_ckpt_to_load = ("chs", "st", "ps")
            path_ckpt = dir_load * "checkpoint_$(epoch).jld2"
            ckpt = PDEHats.load_keys_jld2(path_ckpt, keys_ckpt_to_load)
            chs = ckpt["chs"]
            st = ckpt["st"] |> dev
            ps = ckpt["ps"]
            ps = ComponentArray(ps) |> dev
            model = PDEHats.get_model(chs, name_model, name_data)
            # Data
            obs = get_obs_batch(name_data, seed, idx_NT; T_max=T_max)
            (input, target) = obs
            initial = selectdim(input, 4, 1:1)
            trajectory = cat(initial, target; dims=4)
            @assert trajectory[:, :, :, 1:1, :] == initial
            # Rollout
            errs_dt = map(c_inds_dt) do (t0, t2)
                t1 = t0 + 1
                input_t0 = selectdim(trajectory, 4, t0:t0) |> dev
                target_t1_t2 = selectdim(trajectory, 4, t1:t2) |> dev
                pred_t1_t2 = PDEHats.rollout(model, input_t0, ps, st, dt)
                errs_t1_t2 =
                    loss_smse(target_t1_t2, pred_t1_t2) |> cpu_device()
                errs_cum = dropdims(sum(errs_t1_t2; dims=1); dims=1)
                return errs_cum
            end
            errs_dt_class = map(1:N) do n
                return median(stack(errs_dt)[n, :])
            end
            return errs_dt_class
        end
    # [n, dT, idx, s]
    errs_stack = stack(errs_array)
    ## Hat Matrix
    hats_array =
        map(Iterators.product(1:N, idx_NTs, seeds)) do (b, idx_NT, seed)
            println("Hat Matrix: $(b)")
            println("Hat Matrix: $(idx_NT)")
            println("Hat Matrix: $(seed)")
            hat = get_hat_normed(
                name_model, name_data, epoch, bra_fn, bra_g, seed, idx_NT
            )
            hat_off =
                map(Iterators.product(1:T, 1:N, 1:T, 1:N)) do (t2, n2, t1, n1)
                    if (t1 == t2) && (n2 == n1)
                        return 0.0f0
                    else
                        return hat[t2, n2, t1, n1]
                    end
                end
            hat_on =
                map(Iterators.product(1:T, 1:N, 1:T, 1:N)) do (t2, n2, t1, n1)
                    if (t1 == t2) && (n2 == n1)
                        return hat[t2, n2, t1, n1]
                    else
                        return 0.0f0
                    end
                end
            return sum(hat_off[:, b, :, b]) / sum(hat_on[:, b, :, b])
        end
    ##
    pcs_t = map(1:dT) do dt
        errs_vec = vec(errs_stack[:, dt, :, :])
        hats_vec = vec(hats_array[:, :, :])
        return cor(hats_vec, errs_vec)
    end
    ##
    return pcs_t
end
# ViT[CE]:  -0.65, -0.64, -0.64, -0.64, -0.63, -0.61, -0.61, -0.61, -0.6, -0.6, -0.6, -0.6, -0.59, -0.59, -0.59,-0.59
# UNet[CE]:  -0.57, -0.54, -0.49, -0.45, -0.44, -0.41, -0.39, -0.37, -0.35, -0.33, -0.31, -0.28, -0.21, -0.024, -0.017, -0.019
# ViT[NS]:  -0.8, -0.75, -0.69, -0.65, -0.63, -0.61, -0.61, -0.6, -0.61, -0.61, -0.61, -0.61, -0.62, -0.62, -0.63, -0.64
# UNet[NS]:  -0.52, -0.48, -0.45, -0.43, -0.41, -0.4, -0.39, -0.38, -0.38, -0.39, -0.39, -0.39, -0.39, -0.36, -0.31, -0.26
