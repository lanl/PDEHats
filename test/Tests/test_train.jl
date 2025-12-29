##
function test_train_pass!(name_model::Symbol, name_data::Symbol)
    obj_fn = PDEHats.loss_mse_scaled
    dev = gpu_device()
    backend_autodiff = AutoZygote()
    use_buffer = false
    return test_train_pass!(
        name_model, name_data, obj_fn, dev, backend_autodiff, use_buffer
    )
end
function test_train_pass!(
    name_model::Symbol,
    name_data::Symbol,
    obj_fn::F,
    dev::MLDataDevices.AbstractDevice,
    backend_autodiff::AbstractADType,
    use_buffer::Bool,
) where {F}
    ##
    seed = 0
    chs = 16
    eta = 1.0f-4
    lambda = 1.0f-6
    size_batch = 2
    ## Misc
    rng = Xoshiro(seed)
    opt = AdamW(; eta=eta, lambda=lambda)
    ##
    if Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    ##
    trajectories = PDEHats.get_trajectories(name_data)
    model, ps, st = PDEHats.get_model(rng, chs, name_model, name_data)
    ##
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
    return test_train_pass!(
        loader, model, ps, st, opt, obj_fn, dev, backend_autodiff
    )
end
function test_train_pass!(
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
    ps = ComponentArray(ps)
    ps = ps |> dev
    st = st |> dev
    ##
    state_train = Training.TrainState(model, ps, st, opt)
    PDEHats.train_pass!(
        state_train, obj_fn, loader; backend_autodiff=backend_autodiff
    )
    ##
    return true
end
##
function test_train_val(name_model::Symbol, name_data::Symbol)
    obj_fn = PDEHats.loss_mse_scaled
    dev = gpu_device()
    backend_autodiff = AutoZygote()
    use_buffer = false
    return test_train_val(
        name_model, name_data, obj_fn, dev, backend_autodiff, use_buffer
    )
end
function test_train_val(
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
    epochs = 1
    ## Misc
    rng = Xoshiro(seed)
    opt = AdamW(; eta=eta, lambda=lambda)
    obj_fn = PDEHats.loss_mse_scaled
    dir_save = projectdir("dir_save_test/test_train_val/$(name_data)/")
    ##
    trajectories = PDEHats.get_trajectories(name_data)
    model, ps, st = PDEHats.get_model(rng, chs, name_model, name_data)
    ##
    if Threads.nthreads() > 1
        use_parallel_loading = true
    else
        use_parallel_loading = false
    end
    ## Get Data
    loader_train = DeviceIterator(
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
    loader_val = DeviceIterator(
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
    return test_train_val(
        loader_train,
        loader_val,
        model,
        ps,
        st,
        opt,
        obj_fn,
        dev,
        backend_autodiff;
        dir_save=dir_save,
    )
end
function test_train_val(
    loader_train::DeviceIterator,
    loader_val::DeviceIterator,
    model::AbstractLuxLayer,
    ps,
    st::NamedTuple,
    opt::Optimisers.AbstractRule,
    obj_fn::F,
    dev::MLDataDevices.AbstractDevice,
    backend_autodiff::AutoZygote;
    epochs::Int=2,
    dir_save::String=projectdir("dir_save_test/test_train_val/"),
) where {F}
    ##
    ps = ComponentArray(ps)
    ps = ps |> dev
    st = st |> dev
    state_train = Training.TrainState(model, ps, st, opt)
    ##
    val_fns = [obj_fn]
    ## Train
    PDEHats.train_val!(
        state_train,
        obj_fn,
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
