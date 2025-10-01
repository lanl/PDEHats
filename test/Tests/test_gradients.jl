##
function test_gradients(obj_fn, name_model::Symbol)
    dev = gpu_device()
    backend_autodiff = AutoZygote()
    return test_gradients(obj_fn, dev, backend_autodiff, name_model)
end
function test_gradients(obj_fn, dev, backend_autodiff, name_model::Symbol)
    ##
    chs = 16
    rng = Xoshiro(0)
    eta = 1.0f-4
    lambda = 1.0f-6
    opt = AdamW(; eta=eta, lambda=lambda)
    size_batch = 2
    if Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    ##
    trajectories = PDEHats.get_trajectories("testing")
    L = size(trajectories, 1)
    model, ps, st = PDEHats.get_model(rng, chs, name_model; L=L)
    #
    if backend_autodiff == AutoZygote()
        ps = ComponentArray(ps)
    end
    ps = ps |> dev
    st = st |> dev
    ##
    loader = DeviceIterator(
        dev,
        DataLoader(
            trajectories;
            batchsize=size_batch,
            shuffle=true,
            partial=false,
            buffer=false,
            rng=rng,
            collate=PDEHats.shift_pair,
            parallel=use_parallel_loading,
        ),
    )
    obs = first(loader)
    ##
    state_train = Training.TrainState(model, ps, st, opt);
    ##
    grads, loss, _, state_train = Training.compute_gradients(
        backend_autodiff, obj_fn, obs, state_train
    );
    # Grads
    grads_rms = sqrt(mean(abs2.(getdata(ComponentArray(grads)))))
    println("[$(nameof(obj_fn))] RMS Grads: $grads_rms")
    @assert grads_rms > 0.0f0
    ##
    return true
end
##
