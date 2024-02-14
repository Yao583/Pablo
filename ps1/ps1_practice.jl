using Distributions
#------------------------------------#
#-----------Initialization-----------#
#------------------------------------#

# utility
γ = 2;
beta = 0.97;
w = 5;
r = 0.07;

# productivity shock
sigma_eps = 0.02058;
lambda = 0.99;
m = 1.5;


# number of states 
ne = 15;
nx = 300;
T = 30;

# some matrices
 xgrid = zeros(nx);
 egrid = zeros(ne);
 tempV = zeros(nx*ne);
 tempC = zeros(nx*ne);
 V     = zeros(T,nx,ne);
 C     = zeros(T,nx,ne);
#------------------------------------#
#-----------Grid---------------------#
#------------------------------------#
# savings
 xmax = 4.0;
 xmin = 0.1;
xstep = (xmax - xmin)/(nx-1);
for i = 1:nx
    xgrid[i] = xmin + (i-1)*xstep;
end

# exogenous productivity shocks
sigma_e = sqrt(sigma_eps^2/(1-lambda^2));
estep = 2*m*sigma_e/(ne-1);
for i in 1:ne
    egrid[i] = -m*sigma_e + (i-1)*estep;
end

# Transition probability matrix
 P = zeros(ne,ne);
mm = egrid[2] - egrid[1];
for j in 1:ne
    for k in 1:ne
        if k==1
            P[j,k] = cdf(Normal(), (egrid[k]-lambda*egrid[j]+mm/2)/sigma_eps);
        elseif k==ne
            P[j,k] = 1 - cdf(Normal(),(egrid[k] - lambda*egrid[j] - mm/2)/sigma_eps);
        else
            P[j,k] = cdf(Normal(), (egrid[k]-lambda*egrid[j]+mm/2)/sigma_eps) - cdf(Normal(),(egrid[k] - lambda*egrid[j] - mm/2)/sigma_eps);
        end
    end
end

# take exponential of the egrid
for i in 1:ne
    egrid[i] = exp(egrid[i]);
end


#------------------------------------#
#---------Consumption Computation----#
#------------------------------------#
print("------------------------------")
print("\n")
print("The computation begins", "\n")

using Dates
start = Dates.unix2datetime(time());
for age = T:-1:1
    for ind = 1:nx*ne
    
        ix = convert(Int, ceil(ind/ne));
        ie = convert(Int, floor(mod(ind-0.01,ne))+1);

        cons = 0.0;
        VV = -10^3;

        for ixp in 1:nx
            expected = 0.0;
            if age <T
                for iep in 1:ne
                    expected = expected + P[ie,iep]*V[age+1,ixp,iep];
                end
            end

            cons = (1+r)*xgrid[ix] + egrid[ie]*w - xgrid[ixp];
            if cons < 0
                utility = -10^5;
            end

            utility = cons^(1-γ)/(1-γ) + beta*expected;

            if utility >= VV
                VV = utility;
            end
        end
        V[age,ix,ie] = VV;
        C[age,ix,ie] = cons;
    end

    local finish = convert(Int,Dates.value(Dates.unix2datetime(time()) - start))/1000;
    print("Age:", age, "Time:", finish, "seconds", "\n")
end
finish = convert(Int,Dates.value(Dates.unix2datetime(time()) - start))/1000;
print("The eplased time is:", finish, "seconds", "\n")
print("------------------------------")







