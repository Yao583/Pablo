using NLopt

# the objective function is (1-x)^2 + 2(y - x^2)^2, and the minimizer should be (1,1)
function obj(x::Float64, y::Float64)
    return (1-x)^2 + 2*(y - x^2)^2
end

