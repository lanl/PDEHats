##
struct Benchmark{M,N}
    inputs::AbstractArray{Float32,M}
    targets::AbstractArray{Float32,N}
    preds::AbstractArray{Float32,N}
end
##
include("summary_5pt.jl")
include("Metrics/Metrics.jl")
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
        pred, _ = Lux.apply(model, input, ps, st)
        input_cpu = input |> cpu_device()
        target_cpu = target |> cpu_device()
        pred_cpu = pred |> cpu_device()
        return (input_cpu, target_cpu, pred_cpu)
    end
    ##
    inputs = cat(
        getindex.(itp_batches, 1)...; dims=ndims(getindex(itp_batches[1], 1))
    )
    targets = cat(
        getindex.(itp_batches, 2)...; dims=ndims(getindex(itp_batches[2], 1))
    )
    preds = cat(
        getindex.(itp_batches, 3)...; dims=ndims(getindex(itp_batches[3], 1))
    )
    ##
    bm = Benchmark(inputs, targets, preds)
    return bm
end
