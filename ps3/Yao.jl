using Optimization, OptimizationNLopt, ForwardDiff
# the objective function is (1-x)^2 + 2(y - x^2)^2, and the minimizer should be (1,1)
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 2.0]
f = OptimizationFunction(rosenbrock)
# local minimizer derivative free methods
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [2.0, 2.0])
sol1 = solve(prob, NLopt.LN_NELDERMEAD())

# local minimizer gradient based methods
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 2.0]
f = OptimizationFunction(rosenbrock, AutoForwardDiff())
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [2.0, 2.0])
sol2 = solve(prob, NLopt.LD_LBFGS())

# global minimizer 
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0,p, lb = [-1.0, -1.0], ub = [2.0, 2.0])
sol = solve(prob, NLopt.GN_DIRECT_L(), maxtime = 10.0)

# global minimizer plus local minimizer
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [2.0, 2.0])
sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LN_NELDERMEAD(), maxtime = 10.0,
    local_maxiters = 10)




