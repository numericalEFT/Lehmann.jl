function weightedLeastSqureFit(Gτ, weight, kernel)
    # solve the linear equation: (Kᵀ⋅W⋅K) A = (Kᵀ⋅W) Gτ, where A is the spectral density to calculate
    # assume Gτ: (Nτ, N), kernel: (Nω, Nτ), weight: (Nτ, N)
    # B: (Nω, Nω),  C: (Nω, N)
    if isnothing(weight)
        B = transpose(kernel) * kernel
        C = transpose(transpose(Gτ) * kernel)
        # B = kernel
        # C = Gτ
    else
        @assert size(weight) == size(Gτ)

        W = Diagonal(weight)
        B = kernel * W * transpose(kernel)
        C = kernel * W * Gτ
    end

    kernel = deepcopy(B)
    G = deepcopy(C)

    # ker, ipiv, info = LAPACK.getrf!(B) # LU factorization

    # coeff = LAPACK.getrs!('N', ker, ipiv, C) # LU linear solvor for green=kernel*coeff
    println(B)

    coeff = B \ C #solve green=kernel*coeff

    # println(size(coeff))
    # println(size(kernel))
    # println(size(G))
    # println(size(transpose(coeff) * kernel))
    # println(maximum(abs.(transpose(coeff) * kernel .- G)))
    # exit(0)

    return coeff
end

Gτ = [6.0, 5.0, 7.0, 10.0]
kernel = zeros(4, 2)
kernel[:, 1] = [1.0, 1.0, 1.0, 1.0]
kernel[:, 2] = [1.0, 2.0, 3.0, 4.0]

println(weightedLeastSqureFit(Gτ, nothing, kernel))
