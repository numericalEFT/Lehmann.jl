function _tensor2matrix(tensor, axis)
    # internal function to move the axis dim to the first index, then reshape the tensor into a matrix
    dim = length(size(tensor))
    n1 = size(tensor)[axis]
    partialsize = deleteat!(collect(size(tensor)), axis) # the size of the tensor except the axis-th dimension
    n2 = reduce(*, partialsize)

    if axis == 1 #no need to permutate the axis
        return reshape(tensor, (n1, n2)), partialsize
    elseif axis == 2 && dim == 2 #for matrix, simply transpose, no copy is created
        return transpose(tensor), partialsize
    else
        permu = [i for i = 1:dim]
        permu[1], permu[axis] = axis, 1
        partialsize = collect(size(tensor)[permu][2:end])
        ntensor = permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created 
        # ntensor = nocopy ? PermutedDimsArray(tensor, permu) : permutedims(tensor, permu) # permutate the axis-th and the 1st dim
        ntensor = reshape(ntensor, (n1, n2)) # no copy is created
        return ntensor, partialsize
    end
end

function _matrix2tensor(mat, partialsize, axis)
    # internal function to reshape matrix to a tensor, then swap the first index with the axis-th dimension
    @assert size(mat)[2] == reduce(*, partialsize) # total number of elements of mat and the tensor must match
    tsize = vcat(size(mat)[1], partialsize)
    tensor = reshape(mat, Tuple(tsize))
    dim = length(partialsize) + 1

    if axis == 1
        return tensor
    elseif axis == 2 && dim == 2
        return transpose(tensor) #transpose do not create copy
    else
        permu = [i for i = 1:dim]
        permu[1], permu[axis] = axis, 1
        return permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created
        # ntensor = nocopy ? PermutedDimsArray(tensor, permu) : permutedims(tensor, permu) # permutate the axis-th and the 1st dim
        # return ntensor
    end
end

function _weightedLeastSqureFit(Gτ, error, kernel)
    """
    the current algorithm with weight is only accurate up to 1e-9
    """
    # solve the linear equation: (Kᵀ⋅W⋅K) A = (Kᵀ⋅W) Gτ, where A is the spectral density to calculate, W=1/error^2
    # assume Gτ: (Nτ, N), kernel: (Nω, Nτ), weight: (Nτ, N)
    # B: (Nω, Nω),  C: (Nω, N)
    if isnothing(error)
        B = kernel
        C = Gτ
    else
        @assert size(error) == size(Gτ)
        for i = 1:size(error)[1]
            error[i, :] /= sum(error[i, :]) / length(error[i, :])
        end
        # W = Diagonal(weight)
        # B = transpose(kernel) * W * kernel
        # C = transpose(kernel) * W * Gτ
        w = 1.0 ./ error
        B = w .* kernel
        # B = Diagonal(w) * kernel
        C = w .* Gτ
    end
    # ker, ipiv, info = LAPACK.getrf!(B) # LU factorization
    # coeff = LAPACK.getrs!('N', ker, ipiv, C) # LU linear solvor for green=kernel*coeff
    coeff = B \ C #solve C = B * coeff
    return coeff
end

"""
function tau2dlr(dlrGrid::DLRGrid, green, τGrid = dlrGrid.τ; error = nothing, axis = 1)

    imaginary-time domain to DLR representation

#Members:
- `dlrGrid`: DLRGrid struct.
- `green` : green's function in imaginary-time domain.
- `τGrid` : the imaginary-time grid that Green's function is defined on. 
- `error` : error the Green's function. 
- `axis`: the imaginary-time axis in the data `green`.
"""
function tau2dlr(dlrGrid::DLRGrid, green, τGrid = dlrGrid.τ; error = nothing, axis = 1)
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    @assert size(green)[axis] == length(τGrid)
    ωGrid = dlrGrid.ω

    kernel = Spectral.kernelT(dlrGrid.isFermi, dlrGrid.symmetry, τGrid, ωGrid, dlrGrid.β, true)
    typ = promote_type(eltype(kernel), eltype(green))
    kernel = convert.(typ, kernel)
    green = convert.(typ, green)

    g, partialsize = _tensor2matrix(green, axis)

    if isnothing(error) == false
        @assert size(error) == size(green)
        error, psize = _tensor2matrix(error, axis)
    end
    coeff = _weightedLeastSqureFit(g, error, kernel)
    return _matrix2tensor(coeff, partialsize, axis)
end

"""
function dlr2tau(dlrGrid::DLRGrid, dlrcoeff, τGrid = dlrGrid.τ; axis = 1)

    DLR representation to imaginary-time representation

#Members:
- `dlrGrid` : DLRGrid
- `dlrcoeff` : DLR coefficients
- `τGrid` : expected fine imaginary-time grids 
- `axis`: imaginary-time axis in the data `dlrcoeff`
"""
function dlr2tau(dlrGrid::DLRGrid, dlrcoeff, τGrid = dlrGrid.τ; axis = 1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    @assert size(dlrcoeff)[axis] == size(dlrGrid)

    β = dlrGrid.β
    ωGrid = dlrGrid.ω

    kernel = Spectral.kernelT(dlrGrid.isFermi, dlrGrid.symmetry, τGrid, ωGrid, β, true)

    coeff, partialsize = _tensor2matrix(dlrcoeff, axis)

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    return _matrix2tensor(G, partialsize, axis)
end

"""
function matfreq2dlr(dlrGrid::DLRGrid, green, nGrid = dlrGrid.n; error = nothing, axis = 1)

    Matsubara-frequency representation to DLR representation

#Members:
- `dlrGrid`: DLRGrid struct.
- `green` : green's function in Matsubara-frequency domain
- `nGrid` : the n grid that Green's function is defined on. 
- `error` : error the Green's function. 
- `axis`: the Matsubara-frequency axis in the data `green`
"""
function matfreq2dlr(dlrGrid::DLRGrid, green, nGrid = dlrGrid.n; error = nothing, axis = 1)
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    @assert size(green)[axis] == length(nGrid)
    @assert eltype(nGrid) <: Integer
    ωGrid = dlrGrid.ω

    kernel = Spectral.kernelΩ(dlrGrid.isFermi, dlrGrid.symmetry, nGrid, ωGrid, dlrGrid.β, true)
    typ = promote_type(eltype(kernel), eltype(green))
    kernel = convert.(typ, kernel)
    green = convert.(typ, green)

    g, partialsize = _tensor2matrix(green, axis)

    if isnothing(error) == false
        @assert size(error) == size(green)
        error, psize = _tensor2matrix(error, axis)
    end
    coeff = _weightedLeastSqureFit(g, error, kernel)
    return _matrix2tensor(coeff, partialsize, axis)
end

"""
function dlr2matfreq(dlrGrid::DLRGrid, dlrcoeff, nGrid = dlrGrid.n; axis = 1)

    DLR representation to Matsubara-frequency representation

#Members:
- `dlrGrid` : DLRGrid
- `dlrcoeff` : DLR coefficients
- `nGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `axis`: Matsubara-frequency axis in the data `dlrcoeff`
"""
function dlr2matfreq(dlrGrid::DLRGrid, dlrcoeff, nGrid = dlrGrid.n; axis = 1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    @assert size(dlrcoeff)[axis] == size(dlrGrid)
    @assert eltype(nGrid) <: Integer
    ωGrid = dlrGrid.ω

    kernel = Spectral.kernelΩ(dlrGrid.isFermi, dlrGrid.symmetry, nGrid, ωGrid, dlrGrid.β, true)

    coeff, partialsize = _tensor2matrix(dlrcoeff, axis)

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    return _matrix2tensor(G, partialsize, axis)
end

"""
function tau2matfreq(dlrGrid, green, nNewGrid = dlrGrid.n, τGrid = dlrGrid.τ; error = nothing, axis = 1)

    Fourier transform from imaginary-time to Matsubara-frequency using the DLR representation

#Members:
- `dlrGrid` : DLRGrid
- `green` : green's function in imaginary-time domain
- `nNewGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `τGrid` : the imaginary-time grid that Green's function is defined on. 
- `error` : error the Green's function. 
- `axis`: the imaginary-time axis in the data `green`
"""
function tau2matfreq(dlrGrid, green, nNewGrid = dlrGrid.n, τGrid = dlrGrid.τ; error = nothing, axis = 1)
    coeff = tau2dlr(dlrGrid, green, τGrid; error = error, axis = axis)
    return dlr2matfreq(dlrGrid, coeff, nNewGrid, axis = axis)
end

"""
function matfreq2tau(dlrGrid, green, τNewGrid = dlrGrid.τ, nGrid = dlrGrid.n; error = nothing, axis = 1)

    Fourier transform from Matsubara-frequency to imaginary-time using the DLR representation

#Members:
- `dlrGrid` : DLRGrid
- `green` : green's function in Matsubara-freqeuncy repsentation
- `τNewGrid` : expected fine imaginary-time grids
- `nGrid` : the n grid that Green's function is defined on. 
- `error` : error the Green's function. 
- `axis`: Matsubara-frequency axis in the data `green`
"""
function matfreq2tau(dlrGrid, green, τNewGrid = dlrGrid.τ, nGrid = dlrGrid.n; error = nothing, axis = 1)
    coeff = matfreq2dlr(dlrGrid, green, nGrid; error = error, axis = axis)
    return dlr2tau(dlrGrid, coeff, τNewGrid, axis = axis)
end

"""
function tau2tau(dlrGrid, green, τNewGrid, τGrid = dlrGrid.τ; weight = nothing, axis = 1)

    Interpolation from the old imaginary-time grid to a new grid using the DLR representation

#Members:
- `dlrGrid` : DLRGrid
- `green` : green's function in imaginary-time domain
- `τNewGrid` : expected fine imaginary-time grids
- `τGrid` : the imaginary-time grid that Green's function is defined on. 
- `error` : error the Green's function. 
- `axis`: the imaginary-time axis in the data `green`
"""
function tau2tau(dlrGrid, green, τNewGrid, τGrid = dlrGrid.τ; error = nothing, axis = 1)
    coeff = tau2dlr(dlrGrid, green, τGrid; error = error, axis = axis)
    return dlr2tau(dlrGrid, coeff, τNewGrid, axis = axis)
end

"""
function matfreq2matfreq(dlrGrid, green, nNewGrid, nGrid = dlrGrid.n; error = nothing, axis = 1)

    Fourier transform from Matsubara-frequency to imaginary-time using the DLR representation

#Members:
- `dlrGrid` : DLRGrid
- `green` : green's function in Matsubara-freqeuncy repsentation
- `nNewGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `nGrid` : the n grid that Green's function is defined on. 
- `error` : error the Green's function. 
- `axis`: Matsubara-frequency axis in the data `green`
"""
function matfreq2matfreq(dlrGrid, green, nNewGrid, nGrid = dlrGrid.n; error = nothing, axis = 1)
    coeff = matfreq2dlr(dlrGrid, green, nGrid; error = error, axis = axis)
    return dlr2matfreq(dlrGrid, coeff, nNewGrid, axis = axis)
end

# function convolution(dlrGrid, green1, green2; axis = 1)

# end