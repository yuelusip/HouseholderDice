using LinearAlgebra, HouseholderDice

function sim_ist!(mse::AbstractVector{Float64}, m::Int, n::Int, use_dice::Bool, T::Int, ρ, λ, τ, σₛ, σₙ)

    if use_dice
        # intialize a Householder dice
        Q = dice_gaussian(m, n, 2*T+1)
    else
        # Sample a full Gaussian matrix
        A = randn(m, n)
    end

    # generate the target vector
    ξ = zeros(n)
    for i in 1:n
        # Gaussian-Bernoulli prior
        rand() < ρ && (ξ[i] = randn() * σₛ)
    end

    if use_dice
        y = zeros(m)
        mul_fast!(Q, ξ, y)
    else
        y = A * ξ
    end

    # y = 1/√m A ξ + σₙ * noise
    BLAS.axpby!(σₙ, randn(m), 1/√m, y)

    x = zeros(n)
    yc = zeros(m)
    xc = zeros(n)

    for i in 1:T
        if use_dice
            mul_fast!(Q, x, yc)
            BLAS.axpby!(1.0, y, -1.0/√m, yc)
            mul_fast!(Q, yc, xc, true)
        else
            # The implementation of HouseholderDice is memory efficient. So let's
            # be fair by also using memory-efficient in-place matrix operations here
            mul!(yc, A, x, 1/√m, 0.0)
            BLAS.axpby!(1.0, y, -1.0, yc)
            mul!(xc, A', yc, 1.0, 0.0)
        end

        BLAS.axpy!(τ/√m, xc, x)

        d = 0.0
        # thresholding and calculate MSE
        for j in 1:n
            @inbounds x[j] = max(abs(x[j]) - λ*τ, 0) * sign(x[j])
            d += (x[j] - ξ[j])^2
        end

        mse[i] = d/n
    end

end
