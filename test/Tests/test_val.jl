##
function test_val(val_fn, name_model::Symbol)
    dev = gpu_device()
    backend_autodiff = AutoZygote()
    return test_val(val_fn, dev, backend_autodiff, name_model)
end
##
function test_val(val_fn, dev, backend_autodiff, name_model::Symbol)
    ## Config
    epoch = 1
    epochs = 1
    size_batch = 2
    chs = 16
    seed = 0
    eta = 1.0f-4
    lambda = 1.0f-6
    ## Misc
    rng = Xoshiro(seed)
    opt = AdamW(; eta=eta, lambda=lambda)
    loss_fn = PDEHats.loss_mse_scaled
    val_fns = [val_fn]
    valuations = map(val_fn -> PDEHats.Valuation(val_fn, epochs), val_fns)
    ##
    trajectories = PDEHats.get_trajectories("testing")
    L = size(trajectories, 1)
    model, ps, st = PDEHats.get_model(rng, chs, name_model; L=L)
    if backend_autodiff == AutoZygote()
        ps = ComponentArray(ps)
    end
    ps = ps |> dev
    st = st |> dev
    state_train = Training.TrainState(model, ps, st, opt)
    ## Get Data
    if Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    loader_val = DeviceIterator(
        dev,
        DataLoader(
            trajectories;
            batchsize=size_batch,
            shuffle=false,
            partial=false,
            buffer=false,
            collate=PDEHats.shift_pair,
            parallel=use_parallel_loading,
        ),
    )
    ## Val
    PDEHats.val_pass!(valuations, epoch, state_train, loader_val)
    ##
    return true
end
##
