using DataFrames, CSV, Optim
using GLM, Statistics, StatsBase, LinearAlgebra
using Distributions, Random
using Sobol

function forward_diff(func::Function, x, i = 1)
    # i: default to be 1 to compute scalar differentiation, define specific value otherwise
    # func: function
    # x: input
    # i: index of input with respect of which you want to take derivative of function
    h = max(abs(x[i]), 1)*1.0e-10
    # h = nextfloat(1.0) - 1
    # h = max(abs(x[i]), 1)*nextfloat(0.0)
    dim = length(x)
    dx = zeros(dim,1)
    dx[i] = h 
    # fun(x) = big.(func(x))

    return (func(x.+dx) - func(x))/h
end

function backward_diff(func::Function, x, i = 1)
    # i: default to be 1 to compute scalar differentiation, define specific value otherwise
    # func: function
    # x: input
    # i: index of input with respect of which you want to take derivative of function
    h = max(abs(x[i]), 1)*1.0e-10
    # h = nextfloat(1.0) - 1
    # h = max(abs(x[i]), 1)*nextfloat(0.0)
    dim = length(x)
    dx = zeros(dim,1)
    dx[i] = h 

    return (func(x) - func(x.-dx))/h
end

function centered_diff(func::Function, x, i = 1)
    # i: default to be 1 to compute scalar differentiation, define specific value otherwise
    # func: function
    # x: input
    # i: index of input with respect of which you want to take derivative of function
    h = max(abs(x[i]), 1)*1.0e-10
    dim = length(x)
    dx = zeros(dim,1)
    dx[i] = h 

    return (func(x.+dx) - func(x.-dx))/(2*h)
end

function Richardson_diff(func::Function, x, i = 1)
    # i: default to be 1 to compute scalar differentiation, define specific value otherwise
    # func: function
    # x: input
    # i: index of input with respect of which you want to take derivative of function
    h = max(abs(x[i]), 1)*1.0e-10
    dim = length(x)
    dx = zeros(dim,1)
    dx[i] = h 
    
    return (-func(x.+2*dx) .+ 8*func(x.+dx) .- 8*func(x.-dx) .+ func(x.-2*dx))/(12*h)
end

function complex_diff(func::Function, x, i = 1)
    # i: default to be 1 to compute scalar differentiation, define specific value otherwise
    # func: function
    # x: input
    # i: index of input with respect of which you want to take derivative of function
    h = max(abs(x[i]), 1)*1.0e-10
    dim = length(x)
    dx = zeros(dim,1)
    dx[i] = h 
    
    return imag((func(x.+dx.*im))/h)
    
end

function complex_diff2(func::Function, x, i = 1)
    # i: default to be 1 to compute scalar differentiation, define specific value otherwise
    # func: function
    # x: input
    # i: index of input with respect of which you want to take derivative of function
    h = max(abs(x[i]), 1.0)*1.0e-5
    dim = length(x)
    dx = zeros(dim,1)
    dx[i] = h 
    
    return (func(x) - real(func(x.+dx.*im))) * 2/h^2
    
end

##############################
### Integration ###
##############################

function midpoint_int(func::Function, intv, N = 1000)
    # intv: interval of x to integrate out
    # N: number of subintervals
    beg, fin = intv[1], intv[end]
    h = (fin - beg) / N # length of subinterval

    return sum(func.(beg .+ (float(collect(1:N)) .- 1/2) .* h)) * h
end

function trapezoid_int(func::Function, intv, N = 1000)
    # intv: interval of x to integrate out
    # N: number of subintervals
    beg, fin = intv[1], intv[end]
    h = (fin - beg) / N # length of subinterval

    return (sum(func.(beg .+ (float(collect(1:N-1)) .* h))) + (func(beg) + func(fin))/2) * h
end

function MC_int(func::Function, intv, N = 10000)
    s = SobolSeq(intv[1,:], intv[2,:]) # quasi-random sequence
    p = reduce(hcat, next!(s) for i = 1:N)' # repeat the function?
    return sum(func.(p)) / N
end
