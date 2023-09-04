using Pkg
Pkg.activate("DynLasso")

using LinearAlgebra, Statistics
using OrdinaryDiffEq
using Lux, Zygote#, Enzyme
using Optim, Optimisers
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays # for Component Array
using CairoMakie
using Distributions 

using Infiltrator

# Set a random seed for reproduceable behaviour
using Random
rng = Random.default_rng()
Random.seed!(rng, 000666)

tspan = [0, 20.0]
u0 = [0.0, 1.0]
reltol = 1e-7
abstol = 1e-7
λ = 0.0

####################################################
#############    Real Solution     #################
####################################################

# Expected angular deviation in one unit of time (degrees)
τ₀ = 10.0
ω₀ = 1.0
ω₁ = 2.0

function true_oscilation!(du, u, p, t)
    if t < τ₀
        ω = p[1]
    else 
        ω = p[2]
    end
    du[1] = u[2]
    du[2] = -ω^2 * u[1]
end

prob = ODEProblem(true_oscilation!, u0, tspan, [ω₀, ω₁])
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)
times = true_sol.t

### Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
# X_true = mapslices(x -> rand(sampler(VonMisesFisher(x, κ)), 1), X_noiseless, dims=1)
X_true = X_noiseless


fig = Figure(resolution=(900, 500)) 
ax = CairoMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "Value")

scatter!(ax, times, X_true[1,:], label="Real oscilation")

# Add legend
fig[1, 2] = Legend(fig, ax)
# Save
# save("solution.pdf", fig)

####################################################
###############    Neural ODE    ###################
####################################################

# Normalization of the NN. Ideally we want to do this with L2 norm .
function sigmoid_cap(x)
    min_value = 0.5
    max_value = 2.5
    return min_value + (max_value - min_value) / ( 1.0 + exp(-x) )
end

# Define neural network 
U = Lux.Chain(
    Lux.Dense(1,3,tanh), 
    Lux.Dense(3,3,tanh), 
    Lux.Dense(3,1,sigmoid_cap)
)
p, st = Lux.setup(rng, U)

function ude_oscilation!(du, u, p, t)
    ω = U([t], p, st)[1][1]
    # du[1] .= u[2]
    du .= [u[2], -ω^2 * u[1]]
    nothing
end


####################################################
################    Training   #####################
####################################################

prob_nn = ODEProblem(ude_oscilation!, u0, tspan, p)

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
    l_reg = 0.0
    if λ != 0.0
    times_reg = collect(tspan[1]:5.0:tspan[2])
    for i in 1:(size(times_reg)[1]-1)
        # Solution using AD
    #     # this works, but it is slow!!!
    #     l_reg_t = norm(jacobian(x -> U([x], p, st)[1], t)[1])
        # Discrete solution
        t0 = times_reg[i]
        t1 = times_reg[i+1]
        l_reg_t = norm(U([t1], θ, st)[1] .- U([t0], θ, st)[1])
        l_reg += l_reg_t
    end
    l_reg /= size(times_reg)[1]
end # if
    # @show l_emp, λ * l_reg

    return l_emp + λ * l_reg
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

res1 = Optimization.solve(optprob, ADAM(0.005), callback = callback, maxiters = 4000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 2000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

p_trained = res2.u
u_final = predict(p_trained)

lines!(ax, times, u_final[1,:], label="ODE solution")

save("harmonic_solution.pdf", fig)


fig = Figure(resolution=(900, 500)) 
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"Time", ylabel = "Value")

ωs = reduce(hcat, (t -> U([t], p_trained, st)[1]).(times))

scatter!(ax, times, ωs[1,:], label="ODE parameter")

hlines!(ax, [ω₀, ω₁], 
            xmin=[0,0.5], 
            xmax=[0.5, 1])
vlines!(ax, [τ₀])

fig[1, 2] = Legend(fig, ax)

save("harmonic_parameter.pdf", fig)
