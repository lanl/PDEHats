##
function test_val(name_model::Symbol, name_data::Symbol)
    val_fn = PDEHats.loss_mse_scaled
    dev = gpu_device()
    backend_autodiff = AutoZygote()
    use_buffer = false
    return test_val(
        name_model, name_data, val_fn, dev, backend_autodiff, use_buffer
    )
end
##
function test_val(
    name_model::Symbol,
    name_data::Symbol,
    val_fn::F,
    dev::MLDataDevices.AbstractDevice,
    backend_autodiff::AbstractADType,
    use_buffer::Bool,
) where {F}
    ## Config
    seed = 0
    chs = 16
    eta = 1.0f-4
    lambda = 1.0f-6
    size_batch = 2
    epochs = 1
    ## Misc
    rng = Xoshiro(seed)
    opt = AdamW(; eta=eta, lambda=lambda)
    loss_fn = PDEHats.loss_mse_scaled
    ##
    if Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    ##
    val_fns = [val_fn]
    valuations = map(val_fn -> PDEHats.Valuation(val_fn, epochs), val_fns)
    ##
    trajectories = PDEHats.get_trajectories(name_data)
    model, ps, st = PDEHats.get_model(rng, chs, name_model, name_data)
    ## Get Data
    loader = DeviceIterator(
        dev,
        DataLoader(
            trajectories;
            batchsize=size_batch,
            shuffle=false,
            partial=false,
            buffer=use_buffer,
            collate=PDEHats.shift_pair,
            parallel=use_parallel_loading,
        ),
    )
    ##
    return test_val(
        loader, model, ps, st, opt, valuations, dev, backend_autodiff
    )
end
##
function test_val(
    loader::DeviceIterator,
    model::AbstractLuxLayer,
    ps,
    st::NamedTuple,
    opt::Optimisers.AbstractRule,
    valuations::Vector{<:PDEHats.Valuation},
    dev::MLDataDevices.AbstractDevice,
    backend_autodiff::AutoZygote;
    epoch::Int=1,
)
    ##
    ps = ComponentArray(ps)
    ps = ps |> dev
    st = st |> dev
    state_train = Training.TrainState(model, ps, st, opt)
    ## Val
    PDEHats.val_pass!(valuations, epoch, state_train, loader)
    ##
    return true
end
