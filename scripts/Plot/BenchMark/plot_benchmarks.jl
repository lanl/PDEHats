##
using DrWatson
@quickactivate :PDEHats
#
using Lux, LuxCUDA
using ComponentArrays
using MLUtils
using Random
##
function plot_benchmarks(;
    dev::MLDataDevices.AbstractDevice=gpu_device(),
    dir_saved::String=projectdir("results/July2025/"),
)
    ##
    name_model_array = ["UNet", "ViT"]
    seed_array = [35, 42, 10]
    ##
    for name_model in name_model_array
        for seed in seed_array
            dir_load_model_seed = projectdir(
                dir_saved * "Train/$(name_model)/seed_$(seed)/"
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
##
function plot_benchmarks(
    dir_load::String; dev::MLDataDevices.AbstractDevice=gpu_device()
)
    ## CFG
    dir_load_cfg = join(split(dir_load, "/")[1:(end - 1)], "/") * "/ckpt_0/"
    path_cfg = only(PDEHats.find_files_by_suffix(dir_load_cfg, "cfg.jld2"))
    ##
    keys_cfg = PDEHats.get_keys_jld2(path_cfg)
    keys_cfg_to_load = (
        "ratio_train",
        "ratio_val",
        "seed",
        "chs",
        "name_model",
        "size_batch",
        "T_max",
    )
    cfg = PDEHats.load_keys_jld2(path_cfg, keys_cfg_to_load)
    ##
    @unpack ratio_train, ratio_val, seed, chs, name_model, size_batch, T_max =
        cfg
    T = T_max - 1
    ## CKPT
    path_ckpt = only(PDEHats.find_files_by_suffix(dir_load, "checkpoint.jld2"))
    keys_ckpt_to_load = ("chs", "st", "ps")
    ckpt = PDEHats.load_keys_jld2(path_ckpt, keys_ckpt_to_load)
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
    _, _, dataset_test = PDEHats.get_datasets(
        Lux.replicate(rng), name_data, ratio_train, ratio_val; T_max=T_max
    )
    ## Trajectories
    trajectories_rp = dataset_test.trajectories_rp
    trajectories_crp = dataset_test.trajectories_crp
    trajectories_rpui = dataset_test.trajectories_rpui
    ## Loaders
    loader_rp = DeviceIterator(
        dev,
        DataLoader(
            trajectories_rp;
            batchsize=size_batch,
            shuffle=false,
            partial=false,
            buffer=false,
            parallel=false,
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
            parallel=false,
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
            parallel=false,
            collate=PDEHats.shift_pair,
        ),
    )
    ##
    println("Constructing Benchmarks")
    metrics = [PDEHats.metric_squared_error]
    ## RP
    bm = PDEHats.Benchmark(model, ps, st, loader_rp)
    for metric in metrics
        PDEHats.plot(bm, metric; dir_save=dir_load * "/benchmark/rp/")
    end
    ## CRP
    bm = PDEHats.Benchmark(model, ps, st, loader_crp)
    for metric in metrics
        PDEHats.plot(bm, metric; dir_save=dir_load * "/benchmark/crp/")
    end
    ## RPUI
    bm = PDEHats.Benchmark(model, ps, st, loader_rpui)
    for metric in metrics
        PDEHats.plot(bm, metric; dir_save=dir_load * "/benchmark/rpui/")
    end
    ##
    return nothing
end
