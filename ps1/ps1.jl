
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


## Load the installed packages -----------------------------------------------
using DataFrames, CSV, Optim
using GLM, Statistics, StatsBase, LinearAlgebra
using Distributions, Random
using Distributed, SharedArrays, Base.Threads # for parallel computing
using Plots
#using Parameters, ColorSchemes
#using TexTables
#using Lathe, MLBase

# Enable printing of 10 columns
ENV["COLUMNS"] = 100;

# Define which part you would run 
#part = [2] # 1 or 2 or [1, 2]

## Specify some paths
home = "C:/Users/Jaeyeong Kim"
mac = "/Users/jaeyeongkim"
latex = "$mac/Dropbox/Github/Latex/BC-Courses/[24S-Empirics(Pablo)]/PS/PS1"
wd = "$mac/Dropbox/Github/BC2023/2024-Pablo/ps1 - OLG"

## Set parallel computing
# (1) distributed version
    #addprocs(3)
    #rmprocs(2,3,4)
    #workers()
    # example
    # @distributed for i in 1:10
    #     println("Task on process $(myid()): $i")
    # end
# (2) threads version
    """ 
    @ threads: for multi thread case (multi for loop parallelization)
    @ To change the number of threads from 1 to more, settings.json() and add "julia.NumThreads": 6 and then reopen vs code
    """
    Threads.nthreads()
    Threads.threadid()
    # example
    a = zeros(10)
    Threads.@threads for i = 1:10
        a[i] = Threads.threadid()
    end
    a


## Load functions
include("$wd/funcs.jl")


#------------------------------------------------------------------------#
#### OLG model Value function Derivation from t=1,...,T ####
#------------------------------------------------------------------------#
### OLG model Value function Derivation ---------------------------------#
    # Parameter Settings
    β = 0.96; γ = 1; r = 0.01/4; w = 1; ρ = 0.975; σ = 0.7
    θ = Dict("β" => β, "γ" => γ, "r" => r, "w" => w, "ρ" => ρ, "σ" => σ)
    
    T = 10; na = 5000; ne = 7; a_max = 10; a_min = 0
    grid_θ = Dict("T" => T, "na" => na, "ne" => ne, "a_max" => a_max, "a_min" => a_min)
    
    # Setting Grids for Asset and Productivity and also transition matrix
    a_grid = exponential_grid(a_min, a_max, na);
    e_grid, pi_s, Pi = rouwenhorst(ρ, σ, ne);
    
    # Initialize the value function at T+1
    v_old = zeros(ne, na, T);
    data = zeros(ne, na, T, 3); # V, C, A
    data[:,:,:,1] = v_old;

    # Backward induction from T to 1
    @time V, A_star, C_star = backward_step(Pi, data, e_grid, a_grid, θ, grid_θ);         # Fastest
    #@time V, A_star, C_star = backward_step_vec1(Pi, data, e_grid, a_grid, θ, grid_θ);   # only iep matrix multi: slower
    #@time V, A_star, C_star = backward_step_vec2(Pi, data, e_grid, a_grid, θ, grid_θ);   # iap, iep matrix multi: slower

    # Plot the results
    plot(a_grid, C_star[4,:,1], label="Consumption", xlabel="Asset State", ylabel="Optimal Consumption", title="Optimal Consumption with respect to Asset State")#, xlim=(0,1), ylim=(0,1.1))
    plot(a_grid, A_star[4,:,1], label="Asset", xlabel="Asset State", ylabel="Optimal Asset", title="Optimal Asset Choice with respect to Asset State")#, xlim=(0,1), ylim=(0,1.1))

