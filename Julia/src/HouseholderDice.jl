module HouseholderDice

using LinearAlgebra


"""
    householder!(M, x, old2new)

    Apply a sequence of Householder transformations:
        If old2new = true
            x = H_r ... H_2 H_1 x
        If old2new = false
            x = H_1 H_2 ... H_r x

    Here, H_i = H_i(v_i), where v_i is the i-th column of M and H_i(v_i) is the
    generalized Householder reflector defined in the paper.

    The calculation is done in-place, with the result stored in x.
"""

function householder!(M, x, old2new::Bool)
    n, r = size(M)

    for i in (old2new ? (1 : r) : (r : -1 : 1))
        s = M[i, i] >= 0 ? 1.0 : -1.0

        if i == n
            x[n,1] *= s
        else
            v = view(M, i+1:n, i)
            xv = view(x, i+1:n, 1)

            τ = M[i, i] + s
            c = dot(v, xv)/τ + x[i, 1]

            x[i, 1] = τ * c - s * x[i, 1]

            BLAS.axpby!(c, v, -s, xv)
        end
    end
end

"""
    householder_vec!(M, x, idx)

    Calculate one Householder transformation
        x = H_idx(v) x
    where H_idx(v) is the generalized Householder reflector defined in the paper.

    The calculation is done in-place, with the result stored in x.
"""
@inline @views function householder_vec!(v::AbstractVector{Float64}, x::AbstractVector{Float64}, idx::Int64)
    n = size(v, 1)

    s = v[idx] >= 0 ? 1.0 : -1.0

    if idx == n
        x[n] *= s
    else

        τ = v[idx] + s
        c = dot(v[idx+1:n], x[idx+1:n]) / τ + x[idx]

        x[idx] = τ * c - s * x[idx]

        BLAS.axpby!(c, v[idx+1:n], -s, x[idx+1:n])
    end

    return nothing
end



include("dice_gaussian.jl")
include("dice_ortho.jl")



export dice_gaussian, mul_fast!, dice_ortho, refresh_ensemble!,
 householder!, householder_vec!

end
