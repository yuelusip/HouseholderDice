using HouseholderDice, LinearAlgebra, KrylovKit, StatsBase, QuadGK, Plots, JLD


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

function calc_analytical(zfunc, z_max::Float64)
    function calc_sa_zb(λ,zfunc,a,b)
       v, = quadgk(s-> s^a / (λ-zfunc(s))^b * exp(-s^2/2), -10, 10);
       # display(v)
        return v / sqrt(2*π);
    end

    a, = quadgk(s-> zfunc(s) * s^2 * exp(-s^2/2), -10, 10);
    a /= sqrt(2π);

    θ_list = collect(range(z_max+0.00000000001, z_max+1.5, length=1000));
    s0z1 = calc_sa_zb.(θ_list, zfunc, 0, 1);
    s0z2 = calc_sa_zb.(θ_list, zfunc, 0, 2);
    s2z1 = calc_sa_zb.(θ_list, zfunc, 2, 1);
    s2z2 = calc_sa_zb.(θ_list, zfunc, 2, 2);

    α_ary_dense = s2z1 ./ (s2z1-s0z1);
    μ_list = 1 ./ (θ_list - 1 ./s2z1 .- a);
    L_list = θ_list + (1 ./α_ary_dense .- 1)./s0z1;
    ρ = (1 .+ (1 ./ α_ary_dense .- 1) .* s0z2./(s0z1.^2)) ./ ((1 ./α_ary_dense .- 1) .* s0z2 ./(s0z1.^2) + s2z2 ./(s2z1.^2));

    sel = ρ .< 0;
    ρ[sel] .= 0.0;

    return α_ary_dense, ρ
end


function do_sim(n::Int, nsims::Int, zfunc, α_ary)
    T = 300

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


zfunc(s) = tanh(abs(s))
z_max = 1.0
α_ary = 1.1:0.1:3

n = 10^5
nsims = 1
@time e_ary, ρ_ary_100000, T_ary = do_sim(n, nsims, zfunc, α_ary)

n = 10^3
nsims = 1
e_ary, ρ_ary_1000, T_ary = do_sim(n, nsims, zfunc, α_ary)

α_ary_dense = 1.1:0.02:3
n = 10^3
nsims = 2000
e_ary, ρ_ary_dense_1000, T_ary = do_sim(n, nsims, zfunc, α_ary_dense)

α_analytical, ρ_analytical = calc_analytical(zfunc, z_max)
α_add = collect(1.0:0.01:minimum(α_analytical)-0.001)
α_analytical = [α_add; α_analytical]
ρ_analytical = [zeros(length(α_add)); ρ_analytical]

scatter(α_ary, ρ_ary_1000)
scatter!(α_ary, ρ_ary_100000)
plot!(α_ary_dense, mean(ρ_ary_dense_1000, dims=2))
plot!(α_analytical, ρ_analytical)

fname = string(Base.source_dir(), "/spectral.jld")
save(fname,
    "α_ary", α_ary,
    "α_ary_dense", α_ary_dense,
    "ρ_ary_1000", ρ_ary_1000,
    "ρ_ary_100000", ρ_ary_100000,
    "ρ_ary_dense_1000", ρ_ary_dense_1000,
    "α_analytical", α_analytical,
    "ρ_analytical", ρ_analytical)
