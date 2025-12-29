##
using DrWatson
@quickactivate :PDEHats
#
using Lux, LuxCUDA
using ComponentArrays
using MLUtils
using Random
##
function run_inference(;)
    ##
    dir_saved = projectdir("results/July2025/")
    name_model_array = ["UNet", "ViT"]
    seed_array = [35, 42, 10]
    names_data = ["CE-RP", "CE-CRP", "CE-RPUI"]
    ##
    for name_model in name_model_array
        for seed in seed_array
            for name_data in names_data
                run_inference(name_model, seed, name_data)
            end
        end
    end
    ##
    return nothing
end
##
function run_inference(
    name_model::String,
    seed::Int,
    name_data::String;
    dev::MLDataDevices.AbstractDevice=gpu_device(),
)
    ##
    dir_load_model_seed = last(
        readdir(
            projectdir(dir_saved * "Train/$(name_model)/seed_$(seed)/");
            join=true,
        ),
    )
    ## CKPT
    path_ckpt = only(
        PDEHats.find_files_by_suffix(dir_load_model_seed, "checkpoint.jld2")
    )
    keys_ckpt_to_load = ("chs", "st", "ps")
    ckpt = PDEHats.load_keys_jld2(path_ckpt, keys_ckpt_to_load)
    chs = ckpt["chs"]
    st = ckpt["st"] |> dev
    ps = ckpt["ps"]
    ps = ComponentArray(ps) |> dev
    ## Model
    model = PDEHats.get_model(chs, name_model)
    ## Example Data
    dict_data = load(datadir("data_inference/$(name_data).jld2"))
    input = dict_data["input"]
    target = dict_data["target"]
    ## Inference
    pred, _ = model(input, ps, st)
    ##
    return nothing
end
##
