##
function test_gradients(name_model::Symbol, name_data::Symbol)
    ##
    obj_fn = PDEHats.loss_mse_scaled
    dev = gpu_device()
    backend_autodiff = AutoZygote()
    use_buffer = false
    ##
    return test_gradients(
        name_model, name_data, obj_fn, dev, backend_autodiff, use_buffer
    )
end
function test_gradients(
    name_model::Symbol,
    name_data::Symbol,
    obj_fn::F,
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
    ## Misc
    opt = AdamW(; eta=eta, lambda=lambda)
    rng = Xoshiro(seed)
    ##
    if Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    ##
    trajectories = PDEHats.get_trajectories(name_data)
    model, ps, st = PDEHats.get_model(rng, chs, name_model, name_data)
    ## Get Data
    loader = DeviceIterator(
        dev,
        DataLoader(
            trajectories;
            batchsize=size_batch,
            shuffle=true,
            partial=false,
            buffer=use_buffer,
            rng=rng,
            collate=PDEHats.shift_pair,
            parallel=use_parallel_loading,
        ),
    )
    ##
    return test_gradients(
        loader, model, ps, st, opt, obj_fn, dev, backend_autodiff
    )
end
function test_gradients(
    loader::DeviceIterator,
    model::AbstractLuxLayer,
    ps,
    st::NamedTuple,
    opt::Optimisers.AbstractRule,
    obj_fn::F,
    dev::MLDataDevices.AbstractDevice,
    backend_autodiff::AutoZygote,
) where {F}
    ##
    obs = first(loader)
    ps = ComponentArray(ps)
    ps = ps |> dev
    st = st |> dev
    ##
    state_train = Training.TrainState(model, ps, st, opt)
    ##
    grads, loss, _, state_train = Training.compute_gradients(
        backend_autodiff, obj_fn, obs, state_train
    )
    ## Grads
    grads_rms = sqrt(mean(abs2.(getdata(ComponentArray(grads)))))
    println("[$(nameof(obj_fn))] RMS Grads: $(grads_rms)")
    @assert grads_rms > 0.0f0
    ##
    return true
end
##
