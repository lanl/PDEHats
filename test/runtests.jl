##
using DrWatson, Test
@quickactivate :PDEHats
##
using Lux, LuxCUDA
using ADTypes
using Zygote, ComponentArrays
using Optimisers
using MLUtils, NNlib
using Random, Statistics
using MLDataDevices
##
include(projectdir("test/Tests/Tests.jl"))
## Run test suite
println("Starting tests")
ti = time()
##
for name_data in [:CE_TEST, :NS_TEST]
    for name_model in [:UNet, :ViT]
        @test test_gradients(name_model, name_data)
        @test test_val(name_model, name_data)
        @test test_train_pass!(name_model, name_data)
        @test test_train_val(name_model, name_data)
    end
end
##
ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60; digits=3), " minutes")
