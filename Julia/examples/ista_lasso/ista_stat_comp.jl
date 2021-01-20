using HouseholderDice, LinearAlgebra, Plots, StatsBase

include("sim_ista.jl")

nsims = 200 #For better match, try nsims = 10^4 or 10^5
α = 0.5
n = 10^3
m = Int(round(α * n))

T = 50
ρ = 0.2
τ = 0.3
λ = 2.0
σₛ = 2.0
σₙ = 0.1

mse_dice = zeros(T, nsims)
mse_direct = zeros(T, nsims)

t_dice = 0.0
t_direct = 0.0

for i in 1:nsims
    global t_direct += @elapsed sim_ist!(view(mse_direct, :, i), m, n, false, T, ρ, λ, τ, σₛ, σₙ)
    global t_dice += @elapsed sim_ist!(view(mse_dice, :,i), m, n, true, T, ρ, λ, τ, σₛ, σₙ)
end

plot(1:T, mean(mse_direct, dims=2),label="Direct Simulation")
scatter!(1:T, mean(mse_dice, dims=2), label="Householder Dice")
