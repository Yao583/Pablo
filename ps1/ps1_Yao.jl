using Distributions
using Distributed


#--------------------------------#
#         Initialization         #
#--------------------------------#

# Number of workers
addprocs(2)
workers()

# Grid for savings x in the range [0.1,4.0]
@everywhere nx = 300;
@everywhere xmin = 0.1;
@everywhere xmax = 4.0;

# Grid for exogenous productivity shock e
@everywhere ne = 15;
@everywhere sigma_eps = 0.02058;
@everywhere lambda = 0.99; # close to a random walk
@everywhere m = 15; # these values follow Tauchen (1986)


# Utility function I assume it's CRRA c^(1-γ)/(1-γ)
@everywhere beta = 0.97; # discount factor
@everywhere γ = 2; # risk aversion
@everywhere T = 30; # number of periods

# Prices
@everywhere r  = 0.07; # interest rate
@everywhere w  = 5; # wage

# Initialize grids
@everywhere xgrid = zeros(nx)
@everywhere egrid = zeros(ne)
@everywhere P     = zeros(ne, ne) # transition probability matrix
@everywhere V     = zeros(T, nx, ne) # value function
@everywhere C     = zeros(T, nx, ne) # consumption

# Initialize value function as a shared array
@everywhere using SharedArrays
tempV = SharedArray{Float64,1}(ne*nx, init = tempV -> tempV[localindices(tempV)] = repeat([myid()], length(localindices(tempV))));
tempC = SharedArray{Float64,1}(ne*nx, init = tempC -> tempC[localindices(tempC)] = repeat([myid()], length(localindices(tempC))));
#--------------------------------#
#         Grid creation          #
#--------------------------------#

# Grid for x
size = nx;
xstep = (xmax - xmin) /(size - 1);
for i = 1:nx
  xgrid[i] = xmin + (i-1)*xstep;
end

# Grid for e with Tauchen (1986). Specifically, et = λet-1 + εt, assume εt ~ N(0, sigma_eps^2)
# sigma_eps is sqrt(var(εt)), sigma_e = sqrt(sigma_eps^2/(1-lambda^2)),
# emax = m*sigma_e, emin = - emax,
size = ne;
sigma_e = sqrt((sigma_eps^2) / (1 - (lambda^2)));
estep = (2*sigma_e*m) / (size-1);
for i = 1:ne
  egrid[i] = (-m*sqrt((sigma_eps^2) / (1 - (lambda^2))) + (i-1)*estep);
end

# Transition probability matrix Tauchen (1986)
mm = egrid[2] - egrid[1]; # because the grid is equispaced
for j = 1:ne
  for k = 1:ne
    if(k == 1)
      P[j, k] = cdf(Normal(), (egrid[k] - lambda*egrid[j] + (mm/2))/sigma_eps);
    elseif(k == ne)
      P[j, k] = 1 - cdf(Normal(), (egrid[k] - lambda*egrid[j] - (mm/2))/sigma_eps);
    else
      P[j, k] = cdf(Normal(), (egrid[k] - lambda*egrid[j] + (mm/2))/sigma_eps) - cdf(Normal(), (egrid[k] - lambda*egrid[j] - (mm/2))/sigma_eps);
    end
  end
end

# Exponential of the grid e since ct + xt+1 = (1+r)xt + e^et*w
for i = 1:ne
  egrid[i] = exp(egrid[i]);
end

#--------------------------------#
#     Life-cycle computation     #
#--------------------------------#

print(" \n")
print("Life cycle computation: \n")
print(" \n")
using Dates
start = Dates.unix2datetime(time())
# Initialize value function as a shared array
# tempV = SharedArray{Float64}(ne*nx, init = tempV -> tempV[Base.localindexes(tempV)] = myid())

for age = T:-1:1

  @sync @distributed for ind = 1:(ne*nx)

    ix      = convert(Int, ceil(ind/ne));
    ie      = convert(Int, floor(mod(ind-0.01, ne))+1);

    VV = -10^3;
    CC = 0.0;
    for ixp = 1:nx

      expected = 0.0;
      if(age < T)
        for iep = 1:ne
          expected = expected + P[ie, iep]*V[age+1, ixp, iep];
        end
      end

      cons  = (1 + r)*xgrid[ix] + egrid[ie]*w - xgrid[ixp];
      if cons >= 0
        CC = cons;
      end

      utility = (cons^(1-γ))/(1-γ) + beta*expected;

      if(cons < 0)
        utility = -10^(5);
      end # constraint of positive consumption

      if(utility >= VV)
        VV = utility;
      end

    end

    tempV[ind] = VV;
    tempC[ind] = CC;
  end

  for ind = 1:(ne*nx)

    ix      = convert(Int, ceil(ind/ne));
    ie      = convert(Int, floor(mod(ind-0.01, ne))+1);

    V[age, ix, ie] = tempV[ind]
    C[age, ix, ie] = tempC[ind]
  end

  local finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
  print("Age: ", age, ". Time: ", finish, " seconds. \n")
end

print("\n")
finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
print("TOTAL ELAPSED TIME: ", finish, " seconds. \n")

#---------------------#
#     Some checks     #
#---------------------#

print(" \n")
print(" - - - - - - - - - - - - - - - - - - - - - \n")
print(" \n")
print("The first entries of the value function: \n")
print(" \n")

# I print the first entries of the value function, to check
for i = 1:3
  print(round(V[1, 1, i], digits=5), "\n")
end
print(" \n")
print(" - - - - - - - - - - - - - - - - - - - - - \n")
print(" \n")
print("The first entries of the optimal consumption: \n")
print(" \n")
for i = 1:3
  print(round(C[1,1,i], digits=5), "\n")
end
