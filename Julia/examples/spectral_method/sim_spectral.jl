using HouseholderDice, LinearAlgebra, KrylovKit, StatsBase, Plots


"""
    Implement the matrix-vector multiplication
        yv = A' * diag(z) * A * xv
    using HouseholderDice
"""
function Mx(x0::Vector{Float64}, Q::dice_ortho, z::Vector{Float64}, xv::Vector{Float64},
            yv::Vector{Float64})::Vector{Float64}
    n = length(x0)
    m,  = size(Q.Vl)

    fill!(xv, 0.0)
    copyto!(xv, x0)

    mul_fast!(Q, xv, yv)

    for i = 1:m
        yv[i] *= z[i]
    end

    mul_fast!(Q, yv, xv, true);

    y = Vector{Float64}(undef, n)
    copyto!(y, 1, xv, 1, n)

    return y
end

"""
    Simulate the spectral method. The top eigenvector is calculated by the Krylov-Schur method
"""
function sim_spectral(m::Int, n::Int, T::Int, zfunc, tol::Float64=1e-3,use_dice::Bool=true)
    @assert m > n "We only consider tall orthogonal matrices with m > n"

    ξ = randn(n)
    ξ = ξ * (√m / norm(ξ))

    Q = dice_ortho(m, T)

    # preallocate some workspace
    xv = zeros(m)
    yv = similar(xv)

    copyto!(xv, ξ)

    mul_fast!(Q, xv, yv);
    z = zfunc.(yv);

    f(x) = Mx(x, Q, z, xv, yv)

    x0 = randn(n)
    alg = Lanczos(tol=tol)
    vals, vecs, info = eigsolve(f, randn(n), 1, :LR, alg)

    v = vecs[1]

    ρ = dot(v, ξ)^2 / norm(ξ)^2 / norm(v)^2;

    return (vals[1], ρ, info)
end


"""
    Do the simulation over a number of independent trials
"""
function do_sim(n::Int, nsims::Int, zfunc, α_ary)
    T = 350

    e_ary = zeros(length(α_ary), nsims)
    ρ_ary = similar(e_ary)
    T_ary = similar(e_ary)

    @progress for (i, α) in enumerate(α_ary)
        m = Int(round(α * n))

        for j = 1:nsims
            e, ρ, info = sim_spectral(m, n, T, zfunc)

            e_ary[i, j] = e
            ρ_ary[i, j] = ρ
            T_ary[i, j] = 2 * info.numops + 1
        end
    end

    return e_ary, ρ_ary, T_ary
end

# zfunc specifies the nonlinear function in the generative model
zfunc(s) = tanh(abs(s))
z_max = 1.0
α_ary = 1.1:0.1:3

n = 10^5
nsims = 1 # one trial
@time e_ary, ρ_ary_100000, T_ary1 = do_sim(n, nsims, zfunc, α_ary)

n = 10^3
nsims = 1 # one trial
e_ary, ρ_ary_1000, T_ary = do_sim(n, nsims, zfunc, α_ary)

α_ary_dense = 1.1:0.05:3
n = 10^3
nsims = 10 # 10 trials, and we will show the average
#nsims = 1000
e_ary, ρ_ary_dense_1000, T_ary = do_sim(n, nsims, zfunc, α_ary_dense)

# Calculate the analytical prediction
include("calc_analytical.jl")
α_analytical, ρ_analytical = calc_analytical(zfunc, z_max)
α_add = collect(1.0:0.01:minimum(α_analytical)-0.001)
α_analytical = [α_add; α_analytical]
ρ_analytical = [zeros(length(α_add)); ρ_analytical]

plot(α_analytical, ρ_analytical,linecolor=:black, label="Analytical Prediction", leg=:topleft)
scatter!(α_ary, ρ_ary_100000, label="n = 10^5, 1 trial")
scatter!(α_ary, ρ_ary_1000, label="n = 10^3, 1 trial")
plot!(α_ary_dense, mean(ρ_ary_dense_1000, dims=2), label=string("n = 10^3, ", nsims, " trials"))
xlims!(0.9, maximum(α_ary)+0.2)
