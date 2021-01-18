using LinearAlgebra, Random


"""
    dice_ortho(m, n, T)

    A struct for the Haar-distributed random orthogonal ensemble
"""
mutable struct dice_ortho
    Vl:: Array{Float64,2}
    Tl:: Array{Float64,2}
    Vr:: Array{Float64,2}
    Tr:: Array{Float64,2}
    D:: Array{Float64, 1}
    md:: Int64

    dice_ortho(n::Int64, T::Int64) = 0 < T <= n ? new(zeros(n, T), zeros(T, T), zeros(n, T),
            zeros(T, T), zeros(T), 0) : error("n and T must satisfy 0 < T <= n");
end


"""
    Clear the "memory" of the random orthogonal matrix that has been accumulated so far.
    Reuse the allocated workspace
"""
function refresh_dice!(Q::dice_ortho)
    fill!(Q.Vl, 0.0)
    fill!(Q.Vr, 0.0)
    fill!(Q.Tl, 0.0)
    fill!(Q.Tr, 0.0)
    fill!(Q.D, 1.0)
    Q.md = 0
end

"""
    mul_fast!(Q::dice_ortho, x, y, transQ=false, tol=1e-12)

    Compute matrix-vector multiplications
        y = Q x
    or
        y = Q^T x if transQ = true

    This is done in-place, meaning that the result will be stored in y overwriting its content.
    In addition, the content of Q will also be changed, as it has acquired some new "memory".

    tol: If the norm of a vector is less than tol, we consider it an approximate zero vector.
"""
function mul_fast!(Q::dice_ortho, x::AbstractVecOrMat{Float64}, y::AbstractVecOrMat{Float64},
            transQ::Bool=false, tol::Float64=1e-12)

    n, T = size(Q.Vl)

    # Change to 2-D arrays
    if ndims(x) == 1
        x = reshape(x, :, 1)
    end

    if ndims(y) == 1
        y = reshape(y, :, 1)
    end

    nx, k = size(x)
    ny, ky = size(y)

    @assert nx == n && ny == n "The dimension of x or y does not match the dimension of the matrix."

    @assert k == ky "x and y need to have the same number of columns."

    @assert Q.md + k <= T "Running out of preallocated memory. Using a larger value of T when initializing Q."

    if transQ
        A1 = Q.Vl
        A2 = Q.Vr
    else
        A1 = Q.Vr
        A2 = Q.Vl
    end

    # md is the current memory depth
    md = Q.md

    copy!(y, x)

    for i in 1:k
        if md > 0
            householder!(view(A1, 1:n, 1:md), view(y, :, i:i), true)
        end

        A1v = view(A1, md+1:n, md + 1)
        yv = view(y, md+1:n, i)
        sc = norm(yv)

        if sc < tol
            # In this case, we consider the input x to be approximately living in the space spanned by the right basis vectors.
            # We still increase the memory depth by one by planting one vector, to make the behavior of the code more predicatable.
            A1v[1] = 1.0 # Just a natural basis vector
            sc = 0.0
        else
            BLAS.axpy!(1.0 / sc, yv, A1v)
        end

        A2v = view(A2, md+1:n, md + 1)

        randn!(A2v)

        #tmp = mxcall(:randn, 1, n-md, 1);
        #copy!(A2v, tmp);

        normalize!(A2v)

        copy!(yv, A2v)
        #BLAS.scal!(n-md, sc, yv, 1);

        lmul!(sc, yv)

        if md > 0
            householder!(view(A2, 1:n, 1:md), view(y, :, i:i), false)
        end

        md += 1

    end

    Q.md = md

    return nothing
end
