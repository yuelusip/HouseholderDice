using LinearAlgebra, Random

"""
    dice_gaussian(m, n, T)

    A struct for the iid Gaussian (rectangular Ginibre) ensemble
"""
mutable struct dice_gaussian
    Vl::Array{Float64,2}
    Vr::Array{Float64,2}
    pattern:: Array{Bool,1} # pattern of access
    # Example:
    # pattern = [true, true, false, false, true] -> Ax, Ax, A'x, A'x, Ax

    md::Int64
    σ::Float64

    # internal workspace
    wsl::Array{Float64,1}
    wsr::Array{Float64,1}
    wsc::Array{Float64, 1}

    dice_gaussian(m::Int64, n::Int64, T::Int64) = 0 < T <= min(m, n) ? new(zeros(m, T),
        zeros(n, T), Array{Bool}(undef, T), 0, 1.0, Array{Float64}(undef, m),
        Array{Float64}(undef, n), Array{Float64}(undef, T)) : error("m, n and T must satisfy 0 < T <= min(m, n)");
end



"""
    Clear the "memory" of the Gaussian matrix that has been accumulated so far.
    Reuse the allocated workspace
"""
function refresh_dice!(Q::dice_gaussian)
    fill!(Q.Vl, 0.0)
    fill!(Q.Vr, 0.0)
    Q.md = 0
end



"""
    mul_fast!(Q::dice_gaussian, x, y, transQ=false, tol=1e-12)

    Compute matrix-vector multiplications
        y = Q x
    or
        y = Q^T x if transQ = true

    This is done in-place, meaning that the result will be stored in y overwriting its content.
    In addition, the content of Q will also be changed, as it has acquired some new "memory".

    tol: If the norm of a vector is less than tol, we consider it an approximate zero vector.
"""
@views function mul_fast!(Q::dice_gaussian, x::AbstractVecOrMat{Float64}, y::AbstractVecOrMat{Float64},
            transQ::Bool=false, tol::Float64=1e-12)

    T = size(Q.Vl, 2)

    # Reshape to 2-D arrays
    ndims(x) == 1 && (x = reshape(x, :, 1))
    ndims(y) == 1 && (y = reshape(y, :, 1))

    nx, k = size(x)
    ny, ky = size(y)

    @assert k == ky "x and y need to have the same number of columns."
    @assert Q.md + k <= T "Running out of preallocated memory. Using a larger value of T when initializing Q."

    (A1, ws1, A2, ws2) = transQ ?
                         (Q.Vl, Q.wsl, Q.Vr, Q.wsr) :
                         (Q.Vr, Q.wsr, Q.Vl, Q.wsl)

    n = size(A1, 1)
    m = size(A2, 1)

    @assert nx == n && ny == m "The dimension of x or y needs to match the dimension of the Gaussian matrix."

    md = Q.md

    for i in 1:k
        copyto!(ws1, x[:, i])
        idx = 1

        for j in 1:md
            if transQ ⊻ Q.pattern[j]
                # Householder type
                householder_vec!(A1[:, j], ws1, idx)
                Q.wsc[j] = ws1[idx]
                idx += 1
            else
                # Gaussian type
                Q.wsc[j] = dot(A1[idx:n, j], ws1[idx:n])
            end
        end

        sc = norm(ws1[idx:n])

        if sc < tol
            # In this case, we consider the input x to be approximately living in the space spanned by the right basis vectors.
            # We still increase the memory depth by one by planting one vector, to make the behavior of the code more predicatable.
            A1[idx, md+1] = 1.0 # Just a natural basis vector
            sc = 0.0
        else
            BLAS.axpy!(1.0 / sc, ws1[idx:n], A1[idx:n, md+1])
        end

        idx = md - idx + 2

        randn!(A2[idx:m, md+1])
        #ztmp = mxcall(:randn, 1, m-idx+1, 1);
        #copy!(A2[idx:m, md+1], ztmp)


        fill!(ws2, 0.0)
        BLAS.axpy!(sc, A2[idx:m, md+1], ws2[idx:m])

        for j in md:-1:1
            if transQ ⊻ Q.pattern[j]
                # Gaussian type
                BLAS.axpy!(Q.wsc[j], A2[idx:m, j], ws2[idx:m])
            else
                # Householder type
                idx -= 1
                ws2[idx] = Q.wsc[j]
                householder_vec!(A2[:, j], ws2, idx)
            end
        end

        md += 1
        Q.pattern[md] = !transQ

        copyto!(y[:, i], ws2)
    end

    Q.md = md

    return nothing
end
