
## Directory Setup ----------------------------------------------------------
# root = dirname(@__FILE__)
# cd(root)
# push!(LOAD_PATH, "/Users/jaeyeongkim/Dropbox/Programming/Github/2024-Pablo")
pwd()

## Install packages ---------------------------------------------------------
# using Pkg
# Pkg.add("TexTables")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("Lathe")
# Pkg.add("GLM")
# Pkg.add("StatsPlots")
# Pkg.add("MLBase")
# Pkg.add("KernelDensity")
# Pkg.add("Optim")
# Pkg.add("Distributions")
# Pkg.add("Random")
# Pkg.add("LinearAlgebra")
# Pkg.add("StatsBase")
# # Packages for parallel computing
# Pkg.add("Distributed")
# Pkg.add("SharedArrays")
# Pkg.add("Sobol")

## Load the installed packages -----------------------------------------------
# using DataFrames, CSV
# using Optim
using GLM, Statistics, StatsBase, LinearAlgebra
# using Distributions, Random
using Distributed, SharedArrays, Base.Threads # for parallel computing
# using Plots
# using Parameters, ColorSchemes
# using TexTables
# using Lathe, MLBase
using Sobol # Integration Monte Carlo

# Enable printing of 10 columns
ENV["COLUMNS"] = 100;

# Define which part you would run 
#part = [2] # 1 or 2 or [1, 2]

## Specify some paths
home = "C:/Users/Jaeyeong Kim"
mac = "/Users/jaeyeongkim"
latex = "$mac/Dropbox/Github/Latex/BC-Courses/[24S-Empirics(Pablo)]/PS/PS2"
wd = "$mac/Dropbox/Github/BC2023/2024-Pablo/ps2 - math"

## Set parallel computing
    # # (1) distributed version
    #     addprocs(3)
    #     rmprocs(5,6,7,8,9)
    #     workers()
    #     #example
    #     @distributed for i in 1:10
    #         println("Task on process $(myid()): $i")
    #     end
    # # (2) threads version
    #     """ 
    #     @ threads: for multi thread case (multi for loop parallelization)
    #     @ To change the number of threads from 1 to more, settings.json() and add "julia.NumThreads": 6 and then reopen vs code
    #     """
    #     Threads.nthreads()
    #     Threads.threadid()
    #     # example
    #     a = zeros(10)
    #     Threads.@threads for i = 1:10
    #         a[i] = Threads.threadid()
    #     end
    #     a


## Load functions
include("$wd/funcs.jl")


##############################################################################
## Implementation
##############################################################################


## 1) Differentiation - examples ---------------------------------------------
f(x) = BigFloat.(exp.(x.^2))
x = [2.0]
fwd_diff(f, x, 1::Int64)
bwd_diff(f, x, 1::Int64)
cent_diff(f, x, 1::Int64)
Richardson_extrapol(f, x, 1::Int64)


## 2) Integration - exmaples -------------------------------------------------
#a=1; b=2; n=10000
#f(x) = BigFloat.(x.^2)

midpoint(x->x^2, 1, 2, 10000)
trapezoid(x->x^2, 1, 2, 10000)


## 2) Integration - exmaples -------------------------------------------------
include("$wd/funcs.jl")
#f(x) = exp(x[1])*exp(x[2])
f(x) = sum(x.^2)
a = [0, 0]; b = [1, 1]; k=size(a)[1]
@time MC_crude(f, a, b, 100000)
@time MC_crude(f, a, b, 100000, "sobol")
@time MC_stratify(f, a, b, 0.5, 100000)
@time MC_stratify(f, a, b, 0.5, 100000, "sobol")

# check if variance is reduced in stratified sampling
ns = 1000;
res1 = zeros(ns);res2 = zeros(ns);
@time @distributed for i in 1:ns
    res1[i] = MC_crude(f, a, b, 10000)
    res2[i] = MC_stratify(f, a, b, 0.5, 10000)
end
var(res1)
var(res2)