

## Differentiation related functions
function fwd_diff(f, x, ind=1::Int64)
    size(x)==() && (x = [x])
    h = BigFloat(1e-10) # works only for 77 digits for 1.0+h != 1.0
    xh = BigFloat.(copy(x))
    xh[ind] = xh[ind] + h
    #return (f(xh) - f(x))/h
    return Float64.((f(xh) - f(x))/h)
end

function bwd_diff(f, x, ind=1::Int64)
    size(x)==() && (x = [x])
    h = BigFloat(1e-10) # works only for 77 digits for 1.0+h != 1.0
    xh = BigFloat.(copy(x))
    xh[ind] = xh[ind] - h
    return Float64.((f(x) - f(xh))/h)
end

function cent_diff(f, x, ind=1::Int64)
    size(x)==() && (x = [x])
    h = BigFloat(1e-50) # works only for 77 digits for 1.0+h != 1.0
    xh1 = BigFloat.(copy(x)); xh2 = BigFloat.(copy(x))
    xh1[ind] = xh1[ind] + h; xh2[ind] = xh2[ind] - h
    return Float64.((f(xh1) - f(xh2))/(2*h))
end

x=1

function Richardson_extrapol(f, x, ind=1::Int64)
    size(x)==() && (x = [x])
    h = BigFloat(1e-50) # works only for 77 digits for 1.0+h != 1.0
    xh1 = BigFloat.(copy(x)); xh2 = BigFloat.(copy(x)); xh3 = BigFloat.(copy(x)); xh4 = BigFloat.(copy(x))
    xh1[ind] = xh1[ind] + 2*h; xh2[ind] = xh2[ind] + h;
    xh3[ind] = xh3[ind] - h; xh4[ind] = xh4[ind] - 2*h;
    return Float64.((-f(xh1) + 8*f(xh2) - 8*f(xh3) + f(xh4))/(12*h))
end


## Integration related functions
function midpoint(f, a, b, n::Int64)
    h = BigFloat.((b - a)/n);
    h_vec = a .+ (collect(1:n) .- 1/2).*h;

    xi = BigFloat(a + (b - a) * rand())
    h2 = BigFloat(1e-10)
    dev11 = Richardson_extrapol(f, xi-h2, 1::Int64) 
    dev12 = Richardson_extrapol(f, xi+h2, 1::Int64)
    dev2 = (dev12 - dev11)/(2*h)

    err = (h^2*(b-a)/24)*dev2
    return Float64.(h.*sum(f.(h_vec)).+err)
end


function trapezoid(f, a, b, n::Int64)
    h = BigFloat.((b - a)/n);
    return Float64.(h.*(f.(a)./2 .+ sum(f.(a .+ collect(1:(n-1)).*h)) + f.(b)./2))
end

function MC_crude(f, a, b, n::Int64, type="random")
    k = size(a)[1]
    dif = b .- a

    if type == "sobol"
        sobol = SobolSeq(k)
        x = [next!(sobol) for i in 1:n]
        # Threads.@threads
        @distributed for i in 1:n
            x[i] = copy(x[i]).*dif .+ a
        end
    elseif type == "random"
        x = rand(n, k)
        # Threads.@threads
        @distributed for i in 1:n
            x[i,:] = copy(x[i,:]).*dif .+ a
        end
    end

    return sum(f.(x))/n
end


function MC_stratify(f, a, b, α, n::Int64, type="random")
    n = round(Int, n/2)
    @assert 0< α <1  "α must be within (0, 1)"
    k = size(a)[1]

    diff = b .- a
    diff1 = diff.*α; diff2 = diff.-diff1
    
    if type == "sobol"
        sobol = SobolSeq(k)
        x1 = [next!(sobol) for i in 1:n]
        x2 = [next!(sobol) for i in 1:n]
        # Threads.@threads
        @distributed for i in 1:n
            x1[i] = copy(x1[i]).*diff1 .+ a
            x2[i] = copy(x2[i]).*diff2 .+ (a.+diff1)
        end
    elseif type == "random"
        x1 = rand(n, k); x2 = rand(n, k)
        # Threads.@threads
        @distributed for i in 1:n
            x1[i,:] = copy(x1[i,:]).*diff1 .+ a
            x2[i,:] = copy(x2[i,:]).*diff2 .+ (a.+diff1)
        end
    end
    
    return (α*sum(f.(x1))+(1-α)*sum(f.(x2)))/n
end

