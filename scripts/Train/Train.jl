## Config
Base.@kwdef mutable struct Config
    # Dataset
    T_max::Int = 17
    p::AbstractFloat = 0.5f0
    ratio_train::Float32 = 0.95f0
    ratio_val::Float32 = 0.045f0
    # Architecture
    name_model::Symbol = :UNet
    chs::Int = 24
    # Training
    epochs::Int = 10
    eta::Float32 = 5.0f-4
    lambda::Float32 = 1.0f-6
    # Data
    size_batch::Int = 3
    use_parallel_loading::Bool = false
    # Loss
    loss_fn::Function = PDEHats.loss_mse_scaled
    # Valuation
    val_fns::Vector{<:Function} = [PDEHats.loss_mse]
    # Device
    dev = gpu_device()
    backend_autodiff = AutoZygote()
    # Seed
    seed::Int = rand(1:typemax(Int))
    # Saving
    dir_save::String = "dir_save_default/results/Train/"
end
## Initial Train
function train(epochs::Int, seed::Int, name_model::Symbol, chs::Int)
    ## Saving
    dir_save = projectdir("results/Train/$(name_model)/seed_$(seed)/ckpt_0/")
    ## Train
    train(
        Config(;
            seed=seed,
            chs=chs,
            name_model=name_model,
            dir_save=dir_save,
            epochs=epochs,
        ),
    )
    return nothing
end
function train(seed::Int, name_model::Symbol, chs::Int, ckpt_load::Int)
    dir_load_ckpt = projectdir(
        "results/Train/$(name_model)/seed_$(seed)/ckpt_$(ckpt_load)/"
    )
    dir_load_cfg = projectdir(
        "results/Train/$(name_model)/seed_$(seed)/ckpt_0/"
    )
    train(dir_load_ckpt, dir_load_cfg)
    return nothing
end
##
function train(cfg::Config)
    ## Setup Distributed
    distributed_backend = try
        DistributedUtils.initialize(NCCLBackend)
        DistributedUtils.get_distributed_backend(NCCLBackend)
    catch err
        @error "Could not initialize distributed training. Error: $err"
        nothing
    end
    local_rank = if distributed_backend === nothing
        0
    else
        DistributedUtils.local_rank(distributed_backend)
    end
    total_workers = if distributed_backend === nothing
        1
    else
        DistributedUtils.total_workers(distributed_backend)
    end
    is_distributed = total_workers > 1
    should_log = !is_distributed || local_rank == 0
    PDEHats._println(should_log, "Distributed Workers: $total_workers")
    ## Unpack
    @unpack T_max,
    p,
    ratio_train,
    ratio_val,
    name_model,
    chs,
    epochs,
    eta,
    lambda,
    size_batch,
    use_parallel_loading,
    loss_fn,
    val_fns,
    dev,
    backend_autodiff,
    seed,
    dir_save = cfg
    ## Save CFG
    if should_log
        tagsave(projectdir(dir_save * "cfg.jld2"), struct2dict(cfg))
    end
    ## Seeding
    rng = Xoshiro(seed)
    ## Get Data
    PDEHats._println(should_log, "Loading Data")
    dataset_train, dataset_val, _ = PDEHats.get_datasets(
        Lux.replicate(rng), ratio_train, ratio_val; T_max=T_max, p=p
    )
    PDEHats._println(should_log, "Examples Train: $(length(dataset_train))")
    PDEHats._println(should_log, "Examples Val: $(length(dataset_val))")
    ## Distribute Data
    dataset_train_distributed = DistributedUtils.DistributedDataContainer(
        distributed_backend, dataset_train
    )
    PDEHats._println(should_log, "Distributed: data")
    ## Parallel/Threads
    if use_parallel_loading && Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    PDEHats._println(should_log, "Parallel Loading: $(use_parallel_loading)")
    PDEHats._println(should_log, "Number Threads: $(Threads.nthreads())")
    ## Get DataLoaders
    T = T_max - 1
    loader_train = DeviceIterator(
        dev,
        DataLoader(
            dataset_train_distributed;
            batchsize=size_batch * T,
            shuffle=true,
            partial=false,
            buffer=false,
            rng=Lux.replicate(rng),
            parallel=use_parallel_loading,
        ),
    )
    loader_val = DeviceIterator(
        dev,
        DataLoader(
            dataset_val;
            batchsize=size_batch * T,
            shuffle=false,
            partial=false,
            buffer=false,
            parallel=use_parallel_loading,
        ),
    )
    PDEHats._println(should_log, "Distributed: loaders")
    ## Make TrainState
    model, ps, st = PDEHats.get_model(Lux.replicate(rng), chs, name_model)
    PDEHats._println(
        should_log, "Total Trainable Parameters: $(Lux.parameterlength(ps))"
    )
    if backend_autodiff == AutoZygote()
        ps = ComponentArray(ps)
    end
    ps = ps |> dev
    st = st |> dev
    ps = DistributedUtils.synchronize!!(distributed_backend, ps)
    st = DistributedUtils.synchronize!!(distributed_backend, st)
    PDEHats._println(should_log, "Distributed: ps and st")
    ## Make OptState
    opt_bare = AdamW(; eta=eta, lambda=lambda)
    opt = DistributedUtils.DistributedOptimizer(distributed_backend, opt_bare)
    PDEHats._println(should_log, "Distributed: opt")
    state_train = Training.TrainState(model, ps, st, opt)
    @set! state_train.optimizer_state = DistributedUtils.synchronize!!(
        distributed_backend, state_train.optimizer_state
    )
    PDEHats._println(should_log, "Distributed: opt_state")
    ## Train
    try
        state_train = PDEHats.train_val!(
            state_train,
            loss_fn,
            epochs,
            val_fns,
            loader_train,
            loader_val;
            backend_autodiff=backend_autodiff,
            should_log=should_log,
            dir_save=dir_save,
        )
    catch err
        if should_log
            msg = sprint(showerror, err)
            full = sprint() do io
                return showerror(io, err, catch_backtrace())
            end
            open("error.txt", "w") do io
                return write(io, msg, "\n\n", full)
            end
        end
    end
    ##
    return nothing
end
## Continue Train
function train(dir_load_ckpt::String, dir_load_cfg::String)
    ## Setup Distributed
    distributed_backend = try
        DistributedUtils.initialize(NCCLBackend)
        DistributedUtils.get_distributed_backend(NCCLBackend)
    catch err
        @error "Could not initialize distributed training. Error: $err"
        nothing
    end
    local_rank = if distributed_backend === nothing
        0
    else
        DistributedUtils.local_rank(distributed_backend)
    end
    total_workers = if distributed_backend === nothing
        1
    else
        DistributedUtils.total_workers(distributed_backend)
    end
    is_distributed = total_workers > 1
    should_log = !is_distributed || local_rank == 0
    PDEHats._println(should_log, "Distributed: $is_distributed")
    PDEHats._println(should_log, "Number Workers: $total_workers")
    ## Gather Pathing
    N_ckpt =
        parse(Int, last(split(split(dir_load_ckpt, "/")[end - 1], "_"))) + 1
    dir_save =
        join(split(dir_load_ckpt, "/")[1:(end - 2)], "/") * "/ckpt_$(N_ckpt)/"
    ## Unpack
    PDEHats._println(should_log, "Loading Config")
    cfg = load(only(PDEHats.find_files(dir_load_cfg, "cfg", ".jld2")))
    @unpack T_max,
    p,
    ratio_train,
    ratio_val,
    name_model,
    chs,
    epochs,
    eta,
    lambda,
    size_batch,
    use_parallel_loading,
    loss_fn,
    val_fns,
    dev,
    backend_autodiff,
    seed = cfg
    ## Original Seeding
    rng = Xoshiro(seed)
    ## Get Data
    PDEHats._println(should_log, "Loading Data")
    dataset_train, dataset_val, _ = PDEHats.get_datasets(
        Lux.replicate(rng), ratio_train, ratio_val; T_max=T_max, p=p
    )
    PDEHats._println(should_log, "Examples Train: $(length(dataset_train))")
    PDEHats._println(should_log, "Examples Val: $(length(dataset_val))")
    ## Distribute Data
    dataset_train_distributed = DistributedUtils.DistributedDataContainer(
        distributed_backend, dataset_train
    )
    PDEHats._println(should_log, "Distributed: data")
    ## Parallel/Threads
    if use_parallel_loading && Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    ## Get DataLoaders
    T = T_max - 1
    loader_train = DeviceIterator(
        dev,
        DataLoader(
            dataset_train_distributed;
            batchsize=size_batch * T,
            shuffle=true,
            partial=false,
            buffer=false,
            rng=Lux.replicate(rng),
            parallel=use_parallel_loading,
        ),
    )
    loader_val = DeviceIterator(
        dev,
        DataLoader(
            dataset_val;
            batchsize=size_batch * T,
            shuffle=false,
            partial=false,
            buffer=false,
            parallel=use_parallel_loading,
        ),
    )
    PDEHats._println(should_log, "Distributed: loaders")
    ## Make TrainState
    model = PDEHats.get_model(chs, name_model)
    PDEHats._println(should_log, "Loading Checkpoint")
    path_ckpt = PDEHats.find_files(dir_load_ckpt, "checkpoint", ".jld2")
    ckpt = load(only(path_ckpt))
    st = Lux.trainmode(ckpt["st"])
    ps = ckpt["ps"]
    st_opt_ckpt = ckpt["st_opt"]
    step_ckpt = ckpt["step"]
    if backend_autodiff == AutoZygote()
        ps = ComponentArray(ps)
    end
    PDEHats._println(
        should_log, "Total Trainable Parameters: $(Lux.parameterlength(ps))"
    )
    ps = ps |> dev
    st = st |> dev
    ps = DistributedUtils.synchronize!!(distributed_backend, ps)
    st = DistributedUtils.synchronize!!(distributed_backend, st)
    PDEHats._println(should_log, "Distributed: ps and st")
    ## Make OptState
    opt_bare = AdamW(; eta=eta, lambda=lambda)
    opt = DistributedUtils.DistributedOptimizer(distributed_backend, opt_bare)
    PDEHats._println(should_log, "Distributed: opt")
    state_train = Training.TrainState(model, ps, st, opt)
    PDEHats._println(should_log, "Remembering optimizer state")
    st_opt_ckpt_state = st_opt_ckpt.state |> dev
    @set! state_train.optimizer_state.state = st_opt_ckpt_state
    @set! state_train.step = step_ckpt
    @set! state_train.optimizer_state = DistributedUtils.synchronize!!(
        distributed_backend, state_train.optimizer_state
    )
    PDEHats._println(should_log, "Distributed: opt_state")
    ## Train
    try
        state_train = PDEHats.train_val!(
            state_train,
            loss_fn,
            epochs,
            val_fns,
            loader_train,
            loader_val;
            backend_autodiff=backend_autodiff,
            should_log=should_log,
            dir_save=dir_save,
        )
    catch err
        if should_log
            msg = sprint(showerror, err)
            full = sprint() do io
                return showerror(io, err, catch_backtrace())
            end
            open("error.txt", "w") do io
                return write(io, msg, "\n\n", full)
            end
        end
    end
    ##
    return nothing
end
