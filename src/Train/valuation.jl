##
struct Valuation{F<:Function}
    val_fn::F
    history_val::Vector{Float32}
end
function Valuation(val_fn::F, epochs::Int) where {F<:Function}
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
)
    ## Unpack
    model = state_train.model
    ps = state_train.parameters
    st = Lux.testmode(state_train.states)
    ## Val
    for obs in loader_val
        (input, target) = obs
        pred, st = Lux.apply(model, input, ps, st)
        for v in valuations
            v.history_val[epoch] += v.val_fn(model, ps, st, input, target, pred)
        end
    end
    ## Report
    for v in valuations
        v.history_val[epoch] /= length(loader_val)
        name_val_fn = nameof(v.val_fn)
        _println(
            should_log,
            "[Epoch $(epoch)] [Val $(name_val_fn): $(v.history_val[epoch])]",
        )
    end
    return valuations
end
function plot(valuation::Valuation; dir_save::String="dir_save_default/")
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
    scatter_1x1(
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
function scatter_1x1(
    vals::AbstractVector{Float32},
    title::String,
    label_x::String,
    label_y::String;
    padding_figure::NTuple{4,Int}=(1, 1, 1, 1),
    dir_save="dir_save_default/",
    name_save="scatter_1x1",
)
    ## Log
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure()
        ax = Makie.Axis(
            fig[1, 1];
            title=title,
            xlabel=label_x,
            ylabel=label_y,
            yscale=log10,
        )
        scatter!(ax, vals)
        return current_figure()
    end
    wsave(projectdir(dir_save * "$(name_save)_log.pdf"), fig)
    ## Linear
    fig = with_theme(theme_aps(); figure_padding=padding_figure) do
        fig = Figure()
        ax = Makie.Axis(fig[1, 1]; title=title, xlabel=label_x, ylabel=label_y)
        scatter!(ax, vals)
        return current_figure()
    end
    wsave(projectdir(dir_save * "$(name_save)_linear.pdf"), fig)
    ##
    return nothing
end
