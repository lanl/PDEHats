##
struct Valuation{F<:Function}
    val_fn::F
    history_val::Vector{Float32}
end
function Valuation(val_fn::F, epochs::Int) where {F<:Function}
    history_val = zeros(Float32, epochs) .+ eps(Float32)
    return Valuation(val_fn, history_val)
end
function Valuation(
    val_fn::F, epoch_start::Int, epoch_end::Int
) where {F<:Function}
    epochs = length(epoch_start:epoch_end)
    history_val = zeros(Float32, epochs) .+ eps(Float32)
    return Valuation(val_fn, history_val)
end
##
function val_pass!(
    valuations::Vector{<:Valuation},
    epoch::Int,
    state_train::Training.TrainState,
    loader_val::DeviceIterator;
    should_log::Bool=true,
    epoch_ckpt::Int=0,
)
    ## Unpack
    model = state_train.model
    ps = state_train.parameters
    st = Lux.testmode(state_train.states)
    ##
    length_loader_val = length(loader_val)
    idx_epoch = epoch - epoch_ckpt
    ## Val
    for (input, target) in loader_val
        pred, _ = Lux.apply(model, input, ps, st)
        for v in valuations
            v.history_val[idx_epoch] +=
                v.val_fn(model, ps, st, input, target, pred) / length_loader_val
        end
    end
    ## Report
    for v in valuations
        name_val_fn = nameof(v.val_fn)
        _println(
            should_log,
            "[Epoch $(epoch)] [Val $(name_val_fn): $(v.history_val[idx_epoch])]",
        )
    end
    return valuations
end
function plot(
    valuation::Valuation; dir_save::String=projectdir("dir_save_default/")
)
    ## Unpack
    name_val_fn = string(nameof(valuation.val_fn))
    history_val = valuation.history_val
    ## Save
    dict = Dict("name_val_fn" => name_val_fn, "history_val" => history_val)
    tagsave(
        projectdir(dir_save * "$(name_val_fn)/val-fn=$(name_val_fn).jld2"), dict
    )
    ## Plot
    title = "Optimization Curve (Test Data)"
    label_x = "Epoch"
    label_y = _nameof(name_val_fn)
    name_save = "val-fn=$(name_val_fn)"
    scatter(
        history_val,
        title,
        label_x,
        label_y;
        dir_save=dir_save * "$(name_val_fn)/",
        name_save=name_save,
    )
    ##
    return nothing
end
