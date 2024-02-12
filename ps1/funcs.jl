"""
240209 jaeyeong kim
This file contains the functions that are used in the main file.

1) Functions for Grid settings of state variables
 - 
 -
2) Functions for Value Function Iteration
 - backward_step
 - within_tolerance
 - val_iter


"""

## 0) Ancillary functions
    function stationary(Pi; tol=1e-11, maxit=10000)
        """
        Find invariant distribution of a Markov chain by iteration.
        """
        pi_s = ones(length(Pi[1, :])) / length(Pi[1, :])
        for it in 1:maxit
            pi_new = Pi' * pi_s
            if maximum(abs.(pi_new .- pi_s)) < tol
                break
            end
            pi_s = copy(pi_new)
            it == maxit && throw(ArgumentError("No convergence after $maxit forward iterations!"))
        end
        return pi_s
    end

    function variance(x , pi_s)
        """
        Variance of discretized random variable with support x and pmf pi.
        """
        return sum(pi_s .* (x .- sum(pi_s .* x)).^2)
    end

    # Define within_tolerance to reduce the norm running time for error b/w v_new and v_old
    function within_tolerance(x1, x2, tol)
        # Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2
        # Flattening: Convert 2-dim matrix to 1-dim vector along 1st row-2nd row-...
        y1 = x1'[:]; y2 = x2'[:] #or vec(x1')
        for i in eachindex(y1)
            abs(y1[i] - y2[i]) > tol && return false # return breaks for loop automatically
        end
        return true
    end

## 1) Define functions for grid setting
    
    # Define function for income(endowment) state/prod.shock grid setting
    function exponential_grid(x_min, x_max, num_x, order=2, pivot=1)
        """ Create exponential grid of a given order """
    
        """ recursively compute exp(x + log κ) - κ up to desired order """
        function _transform(x, i)
            f = exp(x + log(pivot)) - pivot
            i < 2 ? (return x_min + f) : (return _transform(f, i - 1))
        end

        """inverse transform, used to figure out boundary"""
        function _inverse_transform(x, i)
            f = log(x + pivot) - log(pivot)
            i < 2 ? (return f) : (return _inverse_transform(f, i - 1))
        end
    
        """uniform grid with maximum set to implement desired x_max"""
        u_max = _inverse_transform(x_max - x_min, order)
        u_grid = range(0, stop=u_max, length=num_x)
    
        return _transform.(u_grid, order)
    end
    

    # Define function for asset state grid setting
    function rouwenhorst(rho, sigma, N=ne)
        """
        Discretize x[t] = ρx[t-1] + ϵ[t] with Rouwenhorst method.
    
        Parameters
        ----------
        rho   : scalar, persistence
        sigma : scalar, unconditional sd of x[t]
        N     : int, number of states in discretized Markov process
    
        Returns
        ----------
        y  : array (N), states proportional to exp(x) s.t. E[y] = 1
        pi : array (N), stationary distribution of discretized process
        Pi : array (N*N), Markov matrix for discretized process
        """
    
        # Parametrize Rouwenhorst for n=2
        p = (1 + rho) / 2
        Pi = [p 1-p; 1-p p]
    
        # Implement recursion to build from n=3 to n=N
        for n in 3:N
            
            P1 = zeros(n, n)
            P2 = zeros(n, n)
            P3 = zeros(n, n)
            P4 = zeros(n, n)
            
            P1[1:end-1, 1:end-1] = p .* Pi
            P2[1:end-1, 2:end] = (1 - p) .* Pi
            P3[2:end, 1:end-1] = (1 - p) .* Pi
            P4[2:end, 2:end] = p .* Pi

            Pi = P1 .+ P2 .+ P3 .+ P4
            Pi[2:end-1, :] ./= 2
        end
    
        # Invariant distribution and scaling
        pi_s = stationary(Pi)
        x = range(-1, 1, length=N)
        x = x .* (sigma / sqrt(variance(x, pi_s)))
        e_grid = exp.(x) ./ sum(pi_s .* exp.(x))
        
        return e_grid, pi_s, Pi
    end
    

    

## 2) Define functions for value function iteration
    """
    Here, it's just a backward induction from T to 1
    Not an usual backward iteration for value function which is for deriving steady state value function
    """
    # Define backward_step to update value function backward
    function backward_step(Pi, data, e_grid, a_grid, θ, grid_θ)
        """ unpack parameters """
        r, γ, β, w= θ["r"], θ["γ"], θ["β"], θ["w"]
        na, ne, T = grid_θ["na"], grid_θ["ne"], grid_θ["T"]
        
        """ unpack data """
        V, C, A = data[:,:,:,1], data[:,:,:,2], data[:,:,:,3]

        """ for loops """
        for age = T:-1:1                            # t = T, t-1, ..., 1 backward
            #println("Time period is $age")
            Threads.@threads for ia = 1:na          # a (asset at t)
                Threads.@threads for ie = 1:ne      # e (prod state at t)
                    #println("Time period is $age, asset is $ia, prod state is $ie")
                    VV = -10^3;                     # initial cutoff for value comparison
                    for iap = 1:na                  # a'(asset at t+1)
                        expected = 0.0;             # initialize expected value to update
                        if age < T                  # if age==T, there is no future state
                            for iep = 1:ne          # e'(prod state at t+1)
                                """ Stack expected value for each future prod state """
                                expected = expected + Pi[ie,iep]*V[iep, iap, age+1];
                            end
                        end
                        
                        cons = (1 + r)*a_grid[ia] + e_grid[ie]*w - a_grid[iap]; # at T, a_grid[iap] would be naturally set 0 since expected=0
                        if cons <= 0 
                            util = -10^5;
                        else
                            γ == 1 ? (u_fun = log(cons)) : (u_fun = cons^(1-γ)/(1-γ))
                            util = u_fun + β*expected;
                        end
                        
                        util >= VV && (VV = util; C[ie, ia, age] = cons; A[ie, ia, age] = a_grid[iap])
                    end # for iap end
                    V[ie, ia, age] = VV;

                end # for ie end
            end # for ia end
        end # for age end
        return V, A, C
    end # function end
    
    function backward_step_vec1(Pi, data, e_grid, a_grid, θ, grid_θ)
        """ unpack parameters """
        r, γ, β, w= θ["r"], θ["γ"], θ["β"], θ["w"]
        na, ne, T = grid_θ["na"], grid_θ["ne"], grid_θ["T"]
        
        """ unpack data """
        V, C, A = data[:,:,:,1], data[:,:,:,2], data[:,:,:,3]

        """ for loops """
        for age = T:-1:1                            # t = T, t-1, ..., 1 backward
            #println("Time period is $age")
            Threads.@threads for ia = 1:na          # a (asset at t)
                Threads.@threads for ie = 1:ne      # e (prod state at t)
                    #println("Time period is $age, asset is $ia, prod state is $ie")
                    VV = -10^3;                     # initial cutoff for value comparison
                    for iap = 1:na                  # a'(asset at t+1)
                        expected = 0.0;             # initialize expected value to update
                        age < T && (expected = expected + Pi[ie,:]'*V[:, iap, age+1]) # (1,ne)*(ne,1) = (1,1)
                        cons = (1 + r)*a_grid[ia] + e_grid[ie]*w - a_grid[iap];
                        if cons <= 0 
                            util = -10^5;
                        else
                            γ == 1 ? (u_fun = log(cons)) : (u_fun = cons^(1-γ)/(1-γ))
                            util = u_fun + β*expected;
                        end
                        
                        util >= VV && (VV = util; C[ie, ia, age] = cons; A[ie, ia, age] = a_grid[iap])
                    end # for iap end
                    V[ie, ia, age] = VV;
                end # for ie end
            end # for ia end
        end # for age end
        return V, A, C
    end # function end

    function backward_step_vec2(Pi, data, e_grid, a_grid, θ, grid_θ)
        """ unpack parameters """
        r, γ, β, w= θ["r"], θ["γ"], θ["β"], θ["w"]
        na, ne, T = grid_θ["na"], grid_θ["ne"], grid_θ["T"]
        
        """ unpack data """
        V, C, A = data[:,:,:,1], data[:,:,:,2], data[:,:,:,3]

        """ for loops """
        for age = T:-1:1                            # t = T, t-1, ..., 1 backward
            #println("Time period is $age")
            Threads.@threads for ia = 1:na          # a (asset at t)
                Threads.@threads for ie = 1:ne      # e (prod state at t)
                    #println("Time period is $age, asset is $ia, prod state is $ie")
                    expected = 0.0;                 # initialize expected value to update
                    age < T && (expected = expected .+ V[:, :, age+1]'*Pi[ie,:]) # (na,ne)*(ne,1) = (na,1)
                    cons = (1 + r)*a_grid[ia] .+ e_grid[ie]*w .- a_grid; #(na,1)
                    
                    γ == 1 ? (u_fun = [i <= 0 ? -10^5 : log(i) for i in cons]) :
                     (u_fun = [i <= 0 ? -10^5 : i^(1-γ)./(1-γ) for i in cons])
                    
                    util = u_fun .+ β.*expected;
                    maxi = argmax(util)
                    V[ie, ia, age], C[ie, ia, age],A[ie, ia, age] = util[maxi], cons[maxi], a_grid[maxi]

                end # for ie end
            end # for ia end
        end # for age end
        return V, A, C
    end # function end


    """ Obsolete in OLG model """
    # Define Function for Value Function Iteration (VFI)
    function OLG_val_iter(v_old, Pi, e_grid, a_grid, θ, grid_θ, max_iter=10000, tol=1e-8)
        # Unpack parameters
        na, ne, T = grid_θ["na"], grid_θ["ne"], grid_θ["T"]

        # Iterate Value function using backward_step
        data = zeros(T, na, ne, 3) # V, C, A
        data[:,:,:,1] = v_old
        for i = 1:max_iter
            V, A, C = backward_step(Pi, data, e_grid, a_grid, θ, grid_θ)
            
            # if converged, break and return v_new
            if within_tolerance(V, v_old, tol)
                println("Value function converged after $i iterations")
                break
            end
            # Update value function if not converged
            v_old = copy(V)
        end
        return V, A, C
    end
