
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


