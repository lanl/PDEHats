##
using DrWatson, Test
@quickactivate :PDEHats
#
using Lux, LuxCUDA
using ADTypes
using Zygote, ComponentArrays
using Optimisers
using MLUtils, NNlib
using Random, Statistics
#
include(projectdir("test/Tests/Tests.jl"))
## Run test suite
println("Starting tests")
ti = time()
## UNet
@testset "Gradient Tests (UNet)" begin
    @test test_gradients(PDEHats.loss_mse, :UNet)
    @test test_gradients(PDEHats.loss_mse_scaled, :UNet)
end
@testset "Val Tests (UNet)" begin
    @test test_val(PDEHats.loss_mse, :UNet)
    @test test_val(PDEHats.loss_mse_scaled, :UNet)
end
@testset "Train Pass Tests (UNet)" begin
    @test test_train_pass!(:UNet)
end
@testset "Train Val Tests (UNet)" begin
    @test test_train_val(:UNet)
end
## ViT
@testset "Gradient Tests (ViT)" begin
    @test test_gradients(PDEHats.loss_mse, :ViT)
    @test test_gradients(PDEHats.loss_mse_scaled, :ViT)
end
@testset "Val Tests (ViT)" begin
    @test test_val(PDEHats.loss_mse, :ViT)
    @test test_val(PDEHats.loss_mse_scaled, :ViT)
end
@testset "Train Pass Tests (ViT)" begin
    @test test_train_pass!(:ViT)
end
@testset "Train Val Tests (ViT)" begin
    @test test_train_val(:ViT)
end
##
ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60; digits=3), " minutes")
