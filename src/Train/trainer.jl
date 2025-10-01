## Generic Training and Valuations Loop
function train_val!(
    state_train::Training.TrainState,
    loss_fn::Function,
    epochs::Int,
    val_fns::Vector{<:Function},
    loader_train::DeviceIterator,
    loader_val::DeviceIterator;
    should_log::Bool=true,
    dir_save::String="dir_save_default/",
    backend_autodiff::AbstractADType=AutoZygote(),
)
    ## Setup
    valuations = map(val_fn -> Valuation(val_fn, epochs), val_fns)
    ## Optimize
    _println(should_log, "Starting Model Training")
    for epoch in 1:epochs
        ## Train
        time_epoch_start = time()
        state_train, loss_train = train_pass!(
            state_train,
            loss_fn,
            loader_train;
            should_log=should_log,
            backend_autodiff=backend_autodiff,
        )
        time_epoch = time() - time_epoch_start
        _println(
            should_log,
            "[Epoch $(epoch)/$(epochs)] [Opt Loss: $(loss_train)] [Time $(time_epoch)]",
        )
        ## Report
        valuations = val_pass!(
            valuations, epoch, state_train, loader_val; should_log=should_log
        )
        ##
    end
    ## Log
    if should_log
        checkpoint(state_train, epochs; dir_save=dir_save)
        for val in valuations
            plot(val; dir_save=dir_save * "val/")
        end
    end
    ##
    return state_train
end
##
function train_pass!(
    state_train::Training.TrainState,
    loss_fn::Function,
    loader_train::DeviceIterator;
    should_log::Bool=true,
    backend_autodiff::AbstractADType=AutoZygote(),
)
    loss_train = 0.0f0
    ## Mini-Batch update
    for (i, obs) in enumerate(loader_train)
        _, loss_train_step, _, state_train = Training.single_train_step!(
            backend_autodiff, loss_fn, obs, state_train
        )
        loss_train += loss_train_step
        ## Loss
        _println(
            should_log,
            "[Mini-batch: $(i)/$(length(loader_train))] [Loss Mean: $(loss_train / i)] [Loss Step: $(loss_train_step)]",
        )
    end
    loss_train /= length(loader_train)
    return state_train, loss_train
end
##
function checkpoint(
    state_train::Training.TrainState,
    epoch::Int;
    dir_save::String="dir_save_default/",
)
    ## Serialize
    ps_cpu = state_train.parameters |> cpu_device()
    st_cpu = Lux.testmode(state_train.states) |> cpu_device()
    st_opt_cpu = state_train.optimizer_state |> cpu_device()
    step = state_train.step
    model = state_train.model
    ## Saving
    dict = Dict(
        "epoch" => epoch,
        "ps" => ps_cpu,
        "st" => st_cpu,
        "st_opt" => st_opt_cpu,
        "step" => step,
        "chs" => model.chs,
    )
    tagsave(projectdir(dir_save * "checkpoint.jld2"), dict)
    ##
    return nothing
end
##
