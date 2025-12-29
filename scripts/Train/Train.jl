## Config
Base.@kwdef mutable struct Config
    # Dataset
    T_max::Int = 17
    ratio_train::Float32 = 0.65f0
    ratio_val::Float32 = 0.05f0
    # Architecture
    name_model::Symbol = :UNet
    chs::Int = 24
    # Training
    epochs::Int = 10
    eta::Float32 = 5.0f-4
    lambda::Float32 = 1.0f-4
    # Data
    name_data::Symbol = :CE
    size_batch::Int = 3
    use_parallel_loading::Bool = false
    use_buffer::Bool = false
    # Loss
    loss_fn::Function = PDEHats.loss_mse_scaled
    # Valuation
    val_fns::Vector{<:Function} = [PDEHats.loss_mse_scaled]
    # Seed
    seed::Int = rand(1:typemax(Int))
    # Saving
    dir_save::String = projectdir("dir_save_default/results/Train/")
end
## Initial Train
function train(
    name_data::Symbol,
    epochs::Int,
    seed::Int,
    name_model::Symbol,
    chs::Int;
    ratio_train::Float32=0.65f0,
    ratio_val::Float32=0.05f0,
)
    ## Saving
    srt = PDEHats.float_to_string(ratio_train)
    srv = PDEHats.float_to_string(ratio_val)
    dir_save = projectdir(
        "results/Train/$(name_data)/$(name_model)/seed_$(seed)_ratiotrain_$(srt)_ratioval_$(srv)/",
    )
    ## Train
    train(
        Config(;
            seed=seed,
            chs=chs,
            name_model=name_model,
            name_data=name_data,
            dir_save=dir_save,
            epochs=epochs,
            ratio_train=ratio_train,
            ratio_val=ratio_val,
        ),
    )
    ##
    return nothing
end
##
function train(
    cfg::Config;
    dev::MLDataDevices.AbstractDevice=gpu_device(),
    backend_autodiff::AbstractADType=AutoZygote(),
)
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
    PDEHats._println(should_log, "Distributed?: $is_distributed")
    PDEHats._println(should_log, "Distributed Workers: $total_workers")
    ## Unpack
    @unpack T_max,
    ratio_train,
    ratio_val,
    name_model,
    chs,
    epochs,
    eta,
    lambda,
    name_data,
    size_batch,
    use_parallel_loading,
    use_buffer,
    loss_fn,
    val_fns,
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
    dataset_train, dataset_val = PDEHats.get_datasets(
        Lux.replicate(rng),
        name_data,
        ratio_train,
        ratio_val;
        T_max=T_max,
        should_log=should_log,
    )
    PDEHats._println(should_log, "Examples Train: $(length(dataset_train))")
    PDEHats._println(should_log, "Examples Val: $(length(dataset_val))")
    ## Distribute Data
    if is_distributed
        dataset_train = DistributedUtils.DistributedDataContainer(
            distributed_backend, dataset_train
        )
        dataset_val = DistributedUtils.DistributedDataContainer(
            distributed_backend, dataset_val
        )
        PDEHats._println(should_log, "Distributed: datasets")
    end
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
            dataset_train;
            batchsize=size_batch * T,
            shuffle=true,
            partial=false,
            buffer=use_buffer,
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
            buffer=use_buffer,
            parallel=use_parallel_loading,
        ),
    )
    ## Get Model
    model, ps, st = PDEHats.get_model(
        Lux.replicate(rng), chs, name_model, name_data
    )
    PDEHats._println(
        should_log, "Total Trainable Parameters: $(Lux.parameterlength(ps))"
    )
    if backend_autodiff == AutoZygote()
        ps = ComponentArray(ps)
    end
    ps = ps |> dev
    st = st |> dev
    if is_distributed
        ps = DistributedUtils.synchronize!!(distributed_backend, ps)
        st = DistributedUtils.synchronize!!(distributed_backend, st)
        PDEHats._println(should_log, "Distributed: ps and st")
    end
    ## Make OptState
    opt = AdamW(; eta=eta, lambda=lambda)
    if is_distributed
        opt = DistributedUtils.DistributedOptimizer(distributed_backend, opt)
        PDEHats._println(should_log, "Distributed: opt")
    end
    ## Make TrainState
    state_train = Training.TrainState(model, ps, st, opt)
    if is_distributed
        @set! state_train.optimizer_state = DistributedUtils.synchronize!!(
            distributed_backend, state_train.optimizer_state
        )
        PDEHats._println(should_log, "Distributed: opt_state")
    end
    ## Train
    try
        state_train = PDEHats.train_val!(
            state_train,
            loss_fn,
            epochs,
            val_fns,
            loader_train,
            loader_val;
            should_log=should_log,
            dir_save=dir_save,
        )
    catch err
        if should_log
            msg = sprint(showerror, err, catch_backtrace())
            open(projectdir(dir_save * "error.txt"), "w") do io
                return write(io, msg)
            end
        end
    end
    ##
    return nothing
end
## Continue Train
function train(
    seed::Int,
    name_model::Symbol,
    chs::Int,
    name_data::Symbol,
    epoch_ckpt::Int,
    epochs::Int,
)
    name_data = :CE
    dir_load = projectdir(
        "results/Train/$(name_data)/$(name_model)/seed_$(seed)/"
    )
    train(dir_load, epoch_ckpt, epochs)
    return nothing
end
##
function train(
    dir_load::String,
    epoch_ckpt::Int,
    epochs::Int;
    dev::MLDataDevices.AbstractDevice=gpu_device(),
    backend_autodiff::AbstractADType=AutoZygote(),
)
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
    ## Unpack
    PDEHats._println(should_log, "Loading Config")
    keys_cfg_to_load = (
        "T_max",
        "ratio_train",
        "ratio_val",
        "name_model",
        "chs",
        "eta",
        "lambda",
        "name_data",
        "size_batch",
        "use_parallel_loading",
        "use_buffer",
        "loss_fn",
        "val_fns",
        "seed",
        "dir_save",
    )
    path_cfg = dir_load * "cfg.jld2"
    cfg = PDEHats.load_keys_jld2(path_cfg, keys_cfg_to_load)
    @unpack T_max,
    ratio_train,
    ratio_val,
    name_model,
    chs,
    eta,
    lambda,
    name_data,
    size_batch,
    use_parallel_loading,
    use_buffer,
    loss_fn,
    val_fns,
    seed,
    dir_save = cfg
    ## Original Seeding
    rng = Xoshiro(seed)
    ## Get Data
    PDEHats._println(should_log, "Loading Data")
    dataset_train, dataset_val = PDEHats.get_datasets(
        Lux.replicate(rng),
        name_data,
        ratio_train,
        ratio_val;
        T_max=T_max,
        should_log=should_log,
    )
    PDEHats._println(should_log, "Examples Train: $(length(dataset_train))")
    PDEHats._println(should_log, "Examples Val: $(length(dataset_val))")
    ## Distribute Data
    if is_distributed
        dataset_train = DistributedUtils.DistributedDataContainer(
            distributed_backend, dataset_train
        )
        dataset_val = DistributedUtils.DistributedDataContainer(
            distributed_backend, dataset_val
        )
        PDEHats._println(should_log, "Distributed: datasets")
    end
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
            dataset_train;
            batchsize=size_batch * T,
            shuffle=true,
            partial=false,
            buffer=use_buffer,
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
            buffer=use_buffer,
            parallel=use_parallel_loading,
        ),
    )
    ## Get Model
    model = PDEHats.get_model(chs, name_model, name_data)
    PDEHats._println(should_log, "Loading Checkpoint")
    keys_ckpt_to_load = ("chs", "st", "ps", "st_opt", "step")
    path_ckpt = dir_load * "checkpoint_$(epoch_ckpt).jld2"
    ckpt = PDEHats.load_keys_jld2(path_ckpt, keys_ckpt_to_load)
    st = Lux.trainmode(ckpt["st"])
    ps = ckpt["ps"]
    st_opt_ckpt = ckpt["st_opt"]
    step_ckpt = ckpt["step"]
    PDEHats._println(
        should_log, "Total Trainable Parameters: $(Lux.parameterlength(ps))"
    )
    if backend_autodiff == AutoZygote()
        ps = ComponentArray(ps)
    end
    ps = ps |> dev
    st = st |> dev
    if is_distributed
        ps = DistributedUtils.synchronize!!(distributed_backend, ps)
        st = DistributedUtils.synchronize!!(distributed_backend, st)
        PDEHats._println(should_log, "Distributed: ps and st")
    end
    ## Make OptState
    opt = AdamW(; eta=eta, lambda=lambda)
    if is_distributed
        opt = DistributedUtils.DistributedOptimizer(distributed_backend, opt)
        PDEHats._println(should_log, "Distributed: opt")
    end
    ## Make TrainState
    state_train = Training.TrainState(model, ps, st, opt)
    PDEHats._println(should_log, "Remembering optimizer state")
    st_opt_ckpt_state = st_opt_ckpt.state |> dev
    @set! state_train.optimizer_state.state = st_opt_ckpt_state
    @set! state_train.step = step_ckpt
    if is_distributed
        @set! state_train.optimizer_state = DistributedUtils.synchronize!!(
            distributed_backend, state_train.optimizer_state
        )
        PDEHats._println(should_log, "Distributed: opt_state")
    end
    ## Train
    try
        state_train = PDEHats.train_val!(
            state_train,
            loss_fn,
            epochs,
            val_fns,
            loader_train,
            loader_val;
            should_log=should_log,
            dir_save=dir_save,
            epoch_ckpt=epoch_ckpt,
        )
    catch err
        if should_log
            msg = sprint(showerror, err, catch_backtrace())
            open(projectdir(dir_save * "error.txt"), "w") do io
                return write(io, msg)
            end
        end
    end
    ##
    return nothing
end
