##
struct Benchmark{M,N}
    inputs::AbstractArray{Float32,M}
    targets::AbstractArray{Float32,N}
    preds::AbstractArray{Float32,N}
end
##
include("summary_5pt.jl")
include("Metrics/Metrics.jl")
include("Figures/Figures.jl")
##
function Benchmark(state_train::Training.TrainState, loader::DeviceIterator)
    model = state_train.model
    ps = state_train.parameters
    st = state_train.states
    return Benchmark(model, ps, st, loader)
end
function Benchmark(d::Dict)
    return Benchmark(d["inputs"], d["targets"], d["preds"])
end
function Benchmark(
    model::AbstractLuxLayer, ps, st::NamedTuple, loader::DeviceIterator
)
    ##
    st = Lux.testmode(st)
    itp_batches = map(loader) do (input, target)
        pred, st = Lux.apply(model, input, ps, st)
        input_cpu = input |> cpu_device()
        target_cpu = target |> cpu_device()
        pred_cpu = pred |> cpu_device()
        return (input_cpu, target_cpu, pred_cpu)
    end
    ##
    inputs = cat(map(itp -> itp[1], itp_batches)...; dims=5)
    targets = cat(map(itp -> itp[2], itp_batches)...; dims=5)
    preds = cat(map(itp -> itp[3], itp_batches)...; dims=5)
    ##
    return Benchmark(inputs, targets, preds)
end
function BenchmarkRollout(
    state_train::Training.TrainState, loader::DeviceIterator
)
    model = state_train.model
    ps = state_train.parameters
    st = state_train.states
    return BenchmarkRollout(model, ps, st, loader)
end
function BenchmarkRollout(
    model::AbstractLuxLayer, ps, st::NamedTuple, loader::DeviceIterator
)
    ##
    st = Lux.testmode(st)
    itp_batches = map(loader) do (input, target)
        pred, st = rollout(model, input, ps, st)
        input_cpu = input |> cpu_device()
        target_cpu = target |> cpu_device()
        pred_cpu = pred |> cpu_device()
        return (input_cpu, target_cpu, pred_cpu)
    end
    ##
    inputs = cat(map(itp -> itp[1], itp_batches)...; dims=5)
    targets = cat(map(itp -> itp[2], itp_batches)...; dims=5)
    preds = cat(map(itp -> itp[3], itp_batches)...; dims=5)
    ##
    return Benchmark(inputs, targets, preds)
end
