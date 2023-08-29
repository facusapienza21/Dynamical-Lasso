
using LinearAlgebra, Statistics
using OrdinaryDiffEq
using Lux, Zygote
using Optim, Optimisers
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays # for Component Array
using CairoMakie


using Infiltrator

# Set a random seed for reproduceable behaviour
using Random
rng = Random.default_rng()
Random.seed!(rng, 000666)

tspan = [0, 130.0]
u0 = [0.0, 0.0, -1.0]
p = 0.1 .* [1.0, 0.0, 0.0]
reltol = 1e-7
abstol = 1e-7

####################################################
#############    Real Solution     #################
####################################################

# Expected angular deviation in one unit of time (degrees)
Δω₀ = 2.0   
# Angular velocity 
ω₀ = Δω₀ * π / 180.0
# Angular momentum
L0 = ω₀ .* [1.0, 0.0, 0.0]

function true_rotation!(du, u, p, t)
    du .= cross(p, u)
end

prob = ODEProblem(true_rotation!, u0, tspan, L0)
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)
X_true = Array(true_sol)
times = true_sol.t

fig = Figure(resolution=(900, 500)) 
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"Stepsize ($\varepsilon$)", ylabel = L"\text{Relative error}")

scatter!(ax, times, (x -> x[2]).(true_sol.u), label="second coordinate")#, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(ax, times, (x -> x[3]).(true_sol.u), label="third coordinate")

# Add legend
fig[1, 2] = Legend(fig, ax)
# Save
# save("solution.pdf", fig)

####################################################
###############    Neural ODE    ###################
####################################################

# Normalization of the NN. Ideally we want to do this with L2 norm .
function sigmoid_cap(x)
    min_value = - 2ω₀
    max_value = + 2ω₀
    return min_value + (max_value - min_value) / ( 1.0 + exp(-x) )
end

# Define neural network 
U = Lux.Chain(
    Lux.Dense(1,5,tanh), 
    Lux.Dense(5,5,tanh), 
    Lux.Dense(5,3,sigmoid_cap)
)
p, st = Lux.setup(rng, U)

function ude_rotation!(du, u, p, t)
    # Angular momentum given by network prediction
    L = U([t], p, st)[1]
    du .= cross(L, u)
    nothing
end

# function loss(p)
#     prob = ODEProblem(ude_rotation!, u0, tspan, p)
#     sol = solve(prob, Tsit5(), 
#                 reltol=reltol, abstol=abstol, 
#                 sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
#     sum((sol.u[end] - [0.0, 0.0, 1.0]).^2)
# end
# @show loss(p)

####################################################
################    Training   #####################
####################################################

prob_nn = ODEProblem(ude_rotation!, u0, tspan, p)

function predict(θ, u0=u0, T=times) 
    _prob = remake(prob_nn, u0 = u0, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Tsit5(), saveat = T,
                abstol = abstol, reltol = reltol,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss(θ)
    u_ = predict(θ)
    # Empirical error
    l_emp = mean(abs2, u_ .- X_true)
    # Regularization
    # here ideally we want the derivative respect to the inpur directly 
    # L_ = U(times, θ, st)[1]
    # l_reg = 
    # sum((u_[end] .- [0.0, 0.0, 1.0]).^2)
    return l_emp
end

losses = Float64[]
callback = function (p, l)
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

res1 = Optimization.solve(optprob, ADAM(), callback = callback, maxiters = 500)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 100)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

p_trained = res2.u
u_final = predict(p_trained)

lines!(ax, times, u_final[2,:], label="ODE second coordinate")
lines!(ax, times, u_final[3,:], label="ODE third coordinate")

save("solution_ODE.pdf", fig)


fig = Figure(resolution=(900, 500)) 
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"Stepsize ($\varepsilon$)", ylabel = L"\text{Relative error}")

hlines!(ax, L0, label="Reference")#, alpha = 0.75, color = :black, label = ["True Data" nothing])

Ls = reduce(hcat, (t -> U([t], p_trained, st)[1]).(times))

scatter!(ax, times, Ls[1,:], label="ODE first coordinate")
scatter!(ax, times, Ls[2,:], label="ODE second coordinate")
scatter!(ax, times, Ls[3,:], label="ODE third coordinate")

save("solution_L.pdf", fig)
