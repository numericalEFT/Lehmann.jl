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
        ntensor = permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created even for axis=1
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
        return permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created even for axis=1
    end
end

"""
function tau2dlr(type, green, dlrGrid::DLRGrid; axis=1, rtol=1e-12)

    imaginary-time domain to DLR representation

#Members:
- `type`: symbol :fermi, :corr, :acorr
- `green` : green's function in imaginary-time domain
- `axis`: the imaginary-time axis in the data `green`
- `rtol`: tolerance absolute error
"""
function tau2dlr(green, dlrGrid::DLRGrid; axis = 1)
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    τGrid = dlrGrid.τ
    ωGrid = dlrGrid.ω

    kernel = Spectral.kernelT(dlrGrid.isFermi, dlrGrid.symmetry, τGrid, ωGrid, dlrGrid.β)
    typ = promote_type(eltype(kernel), eltype(green))
    kernel = convert.(typ, kernel)
    green = convert.(typ, green)
    # kernel, ipiv, info = LAPACK.getrf!(Float64.(kernel)) # LU factorization
    kernel, ipiv, info = LAPACK.getrf!(kernel) # LU factorization

    g, partialsize = _tensor2matrix(green, axis)

    coeff = LAPACK.getrs!('N', kernel, ipiv, g) # LU linear solvor for green=kernel*coeff
    # coeff = kernel \ g #solve green=kernel*coeff
    # println("coeff: ", maximum(abs.(coeff)))

    return _matrix2tensor(coeff, partialsize, axis)
end

"""
function dlr2tau(type, dlrcoeff, dlrGrid::DLRGrid, τGrid; axis=1)

    DLR representation to imaginary-time representation

#Members:
- `type`: symbol :fermi, :corr, :acorr
- `dlrcoeff` : DLR coefficients
- `dlrGrid` : DLRGrid
- `τGrid` : expected fine imaginary-time grids ∈ (0, β]
- `axis`: imaginary-time axis in the data `dlrcoeff`
- `rtol`: tolerance absolute error
"""
function dlr2tau(dlrcoeff, dlrGrid::DLRGrid, τGrid; axis = 1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    # @assert all(τGrid .> 0.0) && all(τGrid .<= dlrGrid.β)
    β = dlrGrid.β
    ωGrid = dlrGrid.ω

    kernel = Spectral.kernelT(dlrGrid.isFermi, dlrGrid.symmetry, τGrid, ωGrid, β)

    coeff, partialsize = _tensor2matrix(dlrcoeff, axis)

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    return _matrix2tensor(G, partialsize, axis)
end

"""
function matfreq2dlr(type, green, dlrGrid::DLRGrid; axis=1, rtol=1e-12)

    Matsubara-frequency representation to DLR representation

#Members:
- `type`: symbol :fermi, :corr, :acorr
- `green` : green's function in Matsubara-frequency domain
- `axis`: the Matsubara-frequency axis in the data `green`
- `rtol`: tolerance absolute error
"""
function matfreq2dlr(green, dlrGrid::DLRGrid; axis = 1)
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    nGrid = dlrGrid.n
    ωGrid = dlrGrid.ω

    kernel = Spectral.kernelΩ(dlrGrid.isFermi, dlrGrid.symmetry, nGrid, ωGrid, dlrGrid.β)
    typ = promote_type(eltype(kernel), eltype(green))
    kernel = convert.(typ, kernel)
    green = convert.(typ, green)
    # kernel, ipiv, info = LAPACK.getrf!(Complex{Float64}.(kernel)) # LU factorization
    kernel, ipiv, info = LAPACK.getrf!(kernel) # LU factorization

    g, partialsize = _tensor2matrix(green, axis)

    coeff = LAPACK.getrs!('N', kernel, ipiv, g) # LU linear solvor for green=kernel*coeff
    # coeff = kernel \ g # solve green=kernel*coeff
    # coeff/=dlrGrid.Euv

    return _matrix2tensor(coeff, partialsize, axis)
end

"""
function dlr2matfreq(type, dlrcoeff, dlrGrid::DLRGrid, nGrid, β=1.0; axis=1)

    DLR representation to Matsubara-frequency representation

#Members:
- `type`: symbol :fermi, :corr, :acorr
- `dlrcoeff` : DLR coefficients
- `dlrGrid` : DLRGrid
- `nGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `axis`: Matsubara-frequency axis in the data `dlrcoeff`
- `rtol`: tolerance absolute error
"""
function dlr2matfreq(dlrcoeff, dlrGrid::DLRGrid, nGrid; axis = 1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    ωGrid = dlrGrid.ω

    kernel = Spectral.kernelΩ(dlrGrid.isFermi, dlrGrid.symmetry, nGrid, ωGrid, dlrGrid.β)

    coeff, partialsize = _tensor2matrix(dlrcoeff, axis)

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    return _matrix2tensor(G, partialsize, axis)
end

"""
function tau2matfreq(type, green, dlrGrid, nGrid; axis=1, rtol=1e-12)

    Fourier transform from imaginary-time to Matsubara-frequency using the DLR representation

#Members:
- `type`: symbol :fermi, :corr, :acorr
- `green` : green's function in imaginary-time domain
- `dlrGrid` : DLRGrid
- `nGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `axis`: the imaginary-time axis in the data `green`
- `rtol`: tolerance absolute error
"""
function tau2matfreq(green, dlrGrid, nGrid; axis = 1)
    coeff = tau2dlr(green, dlrGrid; axis = axis)
    return dlr2matfreq(coeff, dlrGrid, nGrid, axis = axis)
end

"""
function matfreq2tau(type, green, dlrGrid, τGrid; axis=1, rtol=1e-12)

    Fourier transform from Matsubara-frequency to imaginary-time using the DLR representation

#Members:
- `type`: symbol :fermi, :corr, :acorr
- `green` : green's function in Matsubara-freqeuncy repsentation
- `dlrGrid` : DLRGrid
- `τGrid` : expected fine imaginary-time grids
- `axis`: Matsubara-frequency axis in the data `green`
- `rtol`: tolerance absolute error
"""
function matfreq2tau(green, dlrGrid, τGrid; axis = 1)
    coeff = matfreq2dlr(green, dlrGrid; axis = axis)
    return dlr2tau(coeff, dlrGrid, τGrid, axis = axis)
end