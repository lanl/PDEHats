##
using DrWatson
@quickactivate :PDEHats
##
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using DataDeps
##
names_data_CE = ["CE-RP", "CE-RPUI", "CE-CRP"]
batches_data_CE = 0:13
for name_data in names_data_CE
    dir_save = datadir("sim_raw/$(name_data)")
    try
        mkpath(dir_save)
    catch e
    end
    ENV["DATADEPS_LOAD_PATH"] = dir_save
    for batch_data in batches_data_CE
        link_data = "https://huggingface.co/datasets/camlab-ethz/$(name_data)/resolve/main/data_$(batch_data).nc"
        name_datadep = "data_$(batch_data)"
        desc_datadep = "PDEGym dataset ($(name_data)) batch ($(batch_data)), from camlab-ethz Huggingface."
        register(DataDep(name_datadep, desc_datadep, link_data))
        @datadep_str name_datadep
    end
end
##
names_data_NS = ["NS-BB", "NS-Sines", "NS-Gauss"]
batches_data_NS = 0:16
for name_data in names_data_NS
    dir_save = datadir("sim_raw/$(name_data)")
    try
        mkpath(dir_save)
    catch e
    end
    ENV["DATADEPS_LOAD_PATH"] = dir_save
    for batch_data in batches_data_CE
        link_data = "https://huggingface.co/datasets/camlab-ethz/$(name_data)/resolve/main/velocity_$(batch_data).nc"
        name_datadep = "velocity_$(batch_data)"
        desc_datadep = "PDEGym dataset ($(name_data)) batch ($(batch_data)), from camlab-ethz Huggingface."
        register(DataDep(name_datadep, desc_datadep, link_data))
        @datadep_str name_datadep
    end
end
##
