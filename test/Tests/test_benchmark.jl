##
function test_benchmark()
    ##
    trajectories = PDEHats.get_trajectories("testing")
    T = size(trajectories, 4)
    ##
    inputs = selectdim(trajectories, 4, 1:(T - 1))
    targets = selectdim(trajectories, 4, 2:T)
    preds = copy(targets) .* (1.0f0 .+ randn(Float32, size(targets)))
    bm = PDEHats.Benchmark(inputs, preds, targets)
    ##
    metrics = [
        PDEHats.metric_squared_error,
        PDEHats.metric_cons_global,
        PDEHats.metric_cons_global_rollout,
    ]
    ##
    dir_save = projectdir("dir_save_test/benchmark_test/")
    for metric in metrics
        PDEHats.plot(bm, metric; dir_save=dir_save)
    end
    ##
    return true
end
##
