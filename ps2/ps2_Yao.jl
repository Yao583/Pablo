using LinearAlgebra, Statistics
using Sobol, Random

#----------------- Differentiation -----------------#

# forward-differencing
function forward_diff(f::Function, x::Vector{Float64}, i::Int64)
    h = max(abs(x[i]),1)*sqrt(eps());
    dx = zeros(length(x));
    dx[i] = h;
    return (f(x.+dx) - f(x))/h;
end

# centered differencing
function centered_diff(f::Function, x::Vector{Float64}, i::Int64)
    h = max(abs(x[i]),1)*eps()^(1/3);
    dx = zeros(length(x));
    dx[i] = h;
    return (f(x.+dx) - f(x.-dx))/(2*h);
end

# function: y = e^x
f(x) = exp.(x);
x = 1.0;
println("f'(x) = ", exp.([x]), " analytically.");
println("f'(x) = ", forward_diff(f, [x], 1)," using forward differencing.");
println("f'(x) = ", centered_diff(f, [x], 1)," using centered differencing.");

#----------------- Integration -----------------#
# midpoint rule
function midpoint(f::Function, a::Float64, b::Float64, n::Int64)
    h = (b-a)/n;
    return h*sum(f((a .+ collect(1:n) .- 1/2) .*h));
end

# Monte Carlo integration
function monte_carlo(f::Function, a::Float64, b::Float64, n::Int64)
    x = rand(n).*(b-a);
    return sum(f.(x))/n;
end

# stratified Monte Carlo integration
function MC_stratified(f::Function, a::Float64, b::Float64, alpha::Float64, n::Int64)
    diff = b-a;
    x1 = rand(n).*alpha .*diff .+a;
    x2 = b .- rand(n).*alpha .*diff;
    return (alpha*sum(f.(x1)) + (1-alpha)*sum(f.(x2)))/n;
end

# quasi-Monte Carlo integration
function quasi_MC(f::Function, a::Float64, b::Float64, n::Int64)
    sobol = SobolSeq(1);
    x = [next!(sobol) for i in 1:n];
    return sum(f.(x))/n;
end

# function: y = x^3 + 2x^2
f(x) = x.^3 + 2*x.^2;
a = 0.0; b = 1.0;
alpha = 0.5;
n = 10000;
println("∫f(x)dx = ", 11/12, " analytically.");
println("∫f(x)dx = ", midpoint(f,a,b,n), " using midpoint rule.");
println("∫f(x)dx = ", monte_carlo(f,a,b,n), " using Monte Carlo integration.");
println("∫f(x)dx = ", MC_stratified(f,a,b,alpha,n), " using stratified Monte Carlo integration.");
println("∫f(x)dx = ", quasi_MC(f,a,b,n), " using quasi-Monte Carlo integration.");