##
function test_train_pass!(
    name_model::Symbol; backend_autodiff=AutoZygote(), dev=gpu_device()
)
    ##
    chs = 16
    eta = 1.0f-4
    lambda = 1.0f-6
    opt = AdamW(; eta=eta, lambda=lambda)
    seed = 0
    rng = Xoshiro(seed)
    size_batch = 2
    loss_fn = PDEHats.loss_mse_scaled
    if Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    ##
    trajectories = PDEHats.get_trajectories("testing")
    L = size(trajectories, 1)
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
    );
    ##
    model, ps, st = PDEHats.get_model(rng, chs, name_model; L=L)
    if backend_autodiff == AutoZygote()
        ps = ComponentArray(ps)
    end
    ps = ps |> dev;
    st = st |> dev;
    ##
    state_train = Training.TrainState(model, ps, st, opt);
    PDEHats.train_pass!(
        state_train, loss_fn, loader; backend_autodiff=backend_autodiff
    )
    ##
    return true
end
##
function test_train_val(
    name_model::Symbol; dev=gpu_device(), backend_autodiff=AutoZygote()
)
    ## Config
    epochs = 1
    size_batch = 2
    chs = 16
    seed = 0
    eta = 1.0f-4
    lambda = 1.0f-6
    dir_save = projectdir("dir_save_test/train_test/")
    ## Misc
    rng = Xoshiro(seed)
    opt = AdamW(; eta=eta, lambda=lambda)
    loss_fn = PDEHats.loss_mse_scaled
    val_fns = [PDEHats.loss_mse, PDEHats.loss_mse_scaled]
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
    loader_train = DeviceIterator(
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
    ## Train
    PDEHats.train_val!(
        state_train,
        loss_fn,
        epochs,
        val_fns,
        loader_train,
        loader_val;
        dir_save=dir_save,
        backend_autodiff=backend_autodiff,
    )
    ##
    return true
end
##
