##
function test_benchmark(name_data::Symbol)
    ##
    trajectories = PDEHats.get_trajectories(name_data)
    T = size(trajectories, 4)
    ##
    inputs = selectdim(trajectories, 4, 1:(T - 1))
    targets = selectdim(trajectories, 4, 2:T)
    preds = copy(targets) .* (1.0f0 .+ randn(Float32, size(targets)))
    bm = PDEHats.Benchmark(inputs, preds, targets)
    ##
    metrics = [PDEHats.metric_squared_error]
    ##
    dir_save = projectdir("dir_save_test/test_benchmark/$(name_data)/")
    for metric in metrics
        PDEHats.plot(bm, metric; dir_save=dir_save)
    end
    ##
    return true
end
##
