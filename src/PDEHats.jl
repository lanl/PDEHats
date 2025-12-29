module PDEHats
##
using DrWatson
using JLD2
using Lux, LuxCUDA
using AbstractFFTs
using ConcreteStructs
using NNlib, MLUtils
using ChainRulesCore
using ADTypes
using Zygote, ComponentArrays
using Optimisers
using Random
using LinearAlgebra
using StatsBase, Statistics
using Printf
using CairoMakie, MakiePublication, LaTeXStrings
import CairoMakie: plot
using Dates
using NCDatasets
using SharedArrays
## Models
include("Models/Models.jl")
## Losses
include("Losses/Losses.jl")
## Train
include("Train/Train.jl")
## Utils
include("Utils/Utils.jl")
## Figures
include("Figures/Figures.jl")
## Benchmark
# include("Benchmark/Benchmark.jl")
##
end #module
