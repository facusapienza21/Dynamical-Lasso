
using LinearAlgebra
using OrdinaryDiffEq
using Lux
using Lux: Dense, Chain
import Flux.params
using Optim, Optimisers
using Zygote
using SciMLSensitivity

using Infiltrator

# Set a random seed for reproduceable behaviour
using Random
rng = Random.default_rng()
Random.seed!(rng, 000666)

tspan = [0, 130.0]
u0 = [0.0, 0.0, -1.0]
p = 0.1 .* [1.0, 0.0, 0.0]
reltol = 1e-5
abstol = 1e-5

# To do : normilize output of this
U = Chain(
    Dense(1,5,tanh), Dense(5,3)
)
# p = initial_params(U)
p, st = Lux.setup(rng, U)
params = params(p)

function rotation!(du, u, p, t)
    # Angular momentum given by network prediction
    # L  = U(t, p)
    # @infiltrate
    L, _ = Lux.apply(U, [t], p, st)
    # @show L
    du .= cross(L, u)
    nothing
end

function loss(p)
    prob = ODEProblem(rotation!, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)
    sum((sol.u[end] - [0.0, 0.0, 1.0]).^2)
end

gs = gradient(p -> loss(p), params)
gs

# prob = ODEProblem(rotation!, u0, tspan, p)
# sol = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)

# u_final = sol.u[end]


