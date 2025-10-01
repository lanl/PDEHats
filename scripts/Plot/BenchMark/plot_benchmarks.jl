##
using DrWatson
@quickactivate :PDEHats
#
using Lux, LuxCUDA
using ComponentArrays
using MLUtils
using Random
##
function plot_benchmarks(; dev=gpu_device())
    ##
    name_model_array = ["UNet", "ViT"]
    seed_array = [35, 42, 10]
    ##
    for name_model in name_model_array
        for seed in seed_array
            dir_load_model_seed = projectdir(
                "results/Train/$(name_model)/seed_$(seed)/"
            )
            dir_load_model_seed_ckpts = readdir(dir_load_model_seed; join=true)
            for dir_load in dir_load_model_seed_ckpts
                plot_benchmarks(dir_load * "/"; dev=dev)
            end
        end
    end
    ##
    return nothing
end
function plot_benchmarks(dir_load::String; dev=gpu_device())
    ## CFG
    dir_load_cfg = join(split(dir_load, "/")[1:(end - 2)], "/") * "/ckpt_0/"
    cfg_path = only(PDEHats.find_files_by_suffix(dir_load_cfg, "cfg.jld2"))
    cfg = load(cfg_path)
    @unpack ratio_train,
    ratio_val, seed, chs, name_model, size_batch, T_max, p,
    use_parallel_loading = cfg
    T = T_max - 1
    ## CKPT
    path_ckpt = PDEHats.find_files_by_suffix(dir_load, "checkpoint.jld2")
    ckpt = load(only(path_ckpt))
    chs = ckpt["chs"]
    st = ckpt["st"] |> dev
    ps = ckpt["ps"]
    ps = ComponentArray(ps) |> dev
    ## Seeding
    rng = Xoshiro(seed)
    ## Model
    model = PDEHats.get_model(chs, name_model)
    ## Dataset
    println("Loading Data")
    _, dataset_val, _ = PDEHats.get_datasets(
        Lux.replicate(rng), ratio_train, ratio_val; T_max=T_max, p=p
    )
    ## Trajectories
    trajectories_rp = reshape(
        dataset_val.trajectories_rp_r, dataset_val.size_rp
    )
    trajectories_crp = reshape(
        dataset_val.trajectories_crp_r, dataset_val.size_crp
    )
    trajectories_rpui = reshape(
        dataset_val.trajectories_rpui_r, dataset_val.size_rpui
    )
    ## Loaders
    loader_rp = DeviceIterator(
        dev,
        DataLoader(
            trajectories_rp;
            batchsize=size_batch,
            shuffle=false,
            partial=false,
            buffer=false,
            parallel=use_parallel_loading,
            collate=PDEHats.shift_pair,
        ),
    )
    loader_crp = DeviceIterator(
        dev,
        DataLoader(
            trajectories_crp;
            batchsize=size_batch,
            shuffle=false,
            partial=false,
            buffer=false,
            parallel=use_parallel_loading,
            collate=PDEHats.shift_pair,
        ),
    )
    loader_rpui = DeviceIterator(
        dev,
        DataLoader(
            trajectories_rpui;
            batchsize=size_batch,
            shuffle=false,
            partial=false,
            buffer=false,
            parallel=use_parallel_loading,
            collate=PDEHats.shift_pair,
        ),
    )
    ##
    println("Constructing Benchmarks")
    ## RP
    bm = PDEHats.Benchmark(model, ps, st, loader_rp)
    metrics = [PDEHats.metric_squared_error, PDEHats.metric_cons_global]
    for metric in metrics
        PDEHats.plot(bm, metric; dir_save=dir_load * "benchmark/rp/")
    end
    bm_rollout = PDEHats.BenchmarkRollout(model, ps, st, loader_rp)
    metrics_rollout = [
        PDEHats.metric_squared_error, PDEHats.metric_cons_global_rollout
    ]
    for metric in metrics_rollout
        PDEHats.plot(
            bm_rollout, metric; dir_save=dir_load * "benchmark_rollout/rp/"
        )
    end
    ## CRP
    bm = PDEHats.Benchmark(model, ps, st, loader_crp)
    metrics = [PDEHats.metric_squared_error, PDEHats.metric_cons_global]
    for metric in metrics
        PDEHats.plot(bm, metric; dir_save=dir_load * "benchmark/crp/")
    end
    bm_rollout = PDEHats.BenchmarkRollout(model, ps, st, loader_crp)
    metrics_rollout = [
        PDEHats.metric_squared_error, PDEHats.metric_cons_global_rollout
    ]
    for metric in metrics_rollout
        PDEHats.plot(
            bm_rollout, metric; dir_save=dir_load * "benchmark_rollout/crp/"
        )
    end
    ## RPUI
    bm = PDEHats.Benchmark(model, ps, st, loader_rpui)
    metrics = [PDEHats.metric_squared_error, PDEHats.metric_cons_global]
    for metric in metrics
        PDEHats.plot(bm, metric; dir_save=dir_load * "benchmark/rpui/")
    end
    bm_rollout = PDEHats.BenchmarkRollout(model, ps, st, loader_rpui)
    metrics_rollout = [
        PDEHats.metric_squared_error, PDEHats.metric_cons_global_rollout
    ]
    for metric in metrics_rollout
        PDEHats.plot(
            bm_rollout, metric; dir_save=dir_load * "benchmark_rollout/rpui/"
        )
    end
    ##
    return nothing
end
