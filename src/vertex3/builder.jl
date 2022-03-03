using LinearAlgebra, Printf
using StaticArrays
# using GenericLinearAlgebra
using Lehmann

const Float = Float64

### faster, a couple of less digits
using DoubleFloats
# const Float = Double64
const Double = Double64

# similar speed as DoubleFloats
# using MultiFloats
# const Float = Float64x2
# const Double = Float64x2

### a couple of more digits, but slower
# using Quadmath
# const Float = Float128

### 64 digits by default, but a lot more slower
# const Float = BigFloat

abstract type Grid end
abstract type FineMesh end

include("./kernel.jl")
include("./frequency.jl")

mutable struct Basis{D, Grid, Mesh}
    ############    fundamental parameters  ##################
    Λ::Float  # UV energy cutoff * inverse temperature
    rtol::Float # error tolerance

    ###############     DLR grids    ###############################
    N::Int # number of basis
    grid::Vector{Grid} # grid for the basis
    rtol::Vector{Float}  # the relative error achieved by adding the current grid point 

    ###############  linear coefficients for orthognalization #######
    Q::Matrix{Double} # , Q' = R^{-1}, Q*R'= I
    R::Matrix{Double}

    ############ fine mesh #################
    mesh::Mesh

    function Basis{d,Grid, Mesh}(Λ, rtol, projector; sym = 1) where {d,Grid, Mesh}
        _Q = Matrix{Float}(undef, (0, 0))
        _R = similar(_Q)
        mesh = Mesh(Λ, rtol, projector, sym)
        return new{d,Grid, Mesh}(Λ, rtol, 0, [], [], _Q, _R, mesh)
    end
end

function addBasis!(basis::Basis{D, G, M}, projector, grid, verbose) where {D,G,M}
    basis.N += 1
    push!(basis.grid, grid)

    basis.Q, basis.R = GramSchmidt(basis, projector)

    # update the residual on the fine mesh
    updateResidual!(basis, projector)

    # the new rtol achieved by adding the new grid point
    push!(basis.rtol, sqrt(maximum(mesh.residual)))

    verbose && @printf("%3i @$(grid) -> error=%16.8g, Rmin=%16.8g\n", basis.N, basis.rtol[end], basis.R[end, end])
end

functoin addBasisBlock!(basis::Basis{D, G, M}, projector, idx, verbose) where {D, G, M}
    addBasis!(basis, proj, basis.mesh.candidates[idx], verbose)

    ## before set the residual of the selected grid point to be zero, do some check
    residual = sqrt(basis.mesh.residual[idx])
    _norm = basis.R[end, end]
    @assert abs(_norm - residual) < basis.rtol * 100 "inconsistent norm on the grid $(basis.grid[end]) $_norm - $residual = $(_norm-residual)"
    if abs(_norm - residual) > basis.rtol * 10
        @warn("inconsistent norm on the grid $(basis.grid[end]) $_norm - $residual = $(_norm-residual)")
    end

    ## set the residual of the selected grid point to be zero
    basis.mesh.selected[idx] = true
    basis.mesh.residual[idx] = 0 # the selected mesh grid has zero residual

    for grid in mirror(mesh, idx)
        addBasis!(basis, proj, grid, verbose)
    end

    return findmax(basis.mesh.residual)
end

function updateResidual!(basis::Basis{D}, projector) where {D}
    N, rtol, mesh = basis.N, basis.rtol, basis.mesh

    # q = Float.(basis.Q[end, :])
    q = Double.(basis.Q[end, :])

    Threads.@threads for idx in 1:mesh.N
        if mesh.selected[idx] == false
            candidate = mesh.candidates[idx]
            pp = sum(q[j] * projector(mesh, candidate, basis.grid[j]) for j in 1:N)
            _residual = mesh.residual[idx] - pp * pp
            if _residual < 0
                if _residual < -basis.rtol
                    @warn("warning: residual smaller than 0 at $candidate got $(mesh.residual[idx]) - $(pp)^2 = $_residual")
                end
                mesh.residual[idx] = 0
            else
                mesh.residual[idx] = _residual
            end
        end
    end
end

"""
Gram-Schmidt process to the last grid point in basis.grid
"""
function GramSchmidt(basis::Basis{D, G, M}, projector) where {D, G, M}
    _Q = zeros(Double, (basis.N, basis.N))
    _Q[1:end-1, 1:end-1] = basis.Q

    _R = zeros(Double, (basis.N, basis.N))
    _R[1:end-1, 1:end-1] = basis.R
    _Q[end, end] = 1

    newgrid = basis.grid[end]

    for qi in 1:basis.N-1
        # overlap = sum(_Q[qi, j] * projector(Double(basis.Λ), basis.D, basis.grid[j], basis.grid[end]) for j in 1:basis.N)
        overlap = sum(_Q[qi, j] * projector(basis, basis.grid[j], newgrid) for j in 1:basis.N)
        _R[qi, end] = overlap
        _Q[end, :] -= overlap * _Q[qi, :]  # <q, qnew> q
    end

    _norm = projector(basis, newgrid, newgrid) - _R[:, end]' * _R[:, end]
    _norm = sqrt(abs(_norm))
    _R[end, end] = _norm
    _Q[end, :] /= _norm

    return _Q, _R
end

function testOrthgonal(basis::Basis{D}, projector) where {D}
    println("testing orthognalization...")
    KK = zeros(Double, (basis.N, basis.N))
    for (i, g1) in enumerate(basis.grid)
        for (j, g2) in enumerate(basis.grid)
            KK[i, j] = projector(basis, g1, g2)
        end
    end
    maxerr = maximum(abs.(KK - basis.R' * basis.R))
    println("Max overlap matrix R'*R Error: ", maxerr)

    maxerr = maximum(abs.(basis.R * basis.Q' - I))
    println("Max R*R^{-1} Error: ", maxerr)

    II = basis.Q * KK * basis.Q'
    maxerr = maximum(abs.(II - I))
    println("Max Orthognalization Error: ", maxerr)

end

# function testResidual(basis, proj)
#     # residual = [Residual(basis, proj, basis.grid[i, :]) for i in 1:basis.N]
#     # println("Max deviation from zero residual: ", maximum(abs.(residual)))
#     println("Max deviation from zero residual on the DLR grids: ", maximum(abs.(basis.residualFineGrid[basis.gridIdx])))
# end

function QR!(basis::Basis{dim, G, M}, proj; idx = [1, ] , N = 10000, verbose = 0) where {dim, G, M}
    #### add the grid in the idx vector first
    for i in idx
        maxResidual, idx = addBasisBlock!(basis, proj, i, verbose)
    end

    ####### add grids that has the maximum residual
    while sqrt(maxResidual) > rtol && basis.N < N

        maxResidual, idx = addBasisBlock!(basis, proj, i, verbose)

        # plotResidual(basis)
        # testOrthgonal(basis)
    end
    testOrthgonal(basis, proj)
    @printf("residual = %.16e\n", maxResidual)
    # plotResidual(basis)
    # plotResidual(basis, proj, Float(0), Float(100), candidate, residual)
    return basis
end

if abspath(PROGRAM_FILE) == @__FILE__

    dim = 2
    basis = Basis{dim, FreqGrid, FreqFineMesh}(10, 1e-6, proj2D, sym = 0)
    QR!(basis, proj2D, verbose = 0)

    basis = Basis{dim, FreqGrid, FreqFineMesh}(100, 1e-8, proj2D, sym = 0)
    @time QR!(basis, proj2D, verbose = 1)
end

# function plotResidual(basis)
#     z = Float64.(basis.residualFineGrid)
#     z = reshape(z, (basis.Nfine, basis.Nfine))
#     # contourf(z)
#     # println(basis.fineGrid)
#     # println(basis.grid)
#     # p = heatmap(Float64.(basis.fineGrid), Float64.(basis.fineGrid), z, xaxis = :log, yaxis = :log)
#     p = heatmap(Float64.(basis.fineGrid), Float64.(basis.fineGrid), z)
#     # p = heatmap(z)
#     x = [basis.grid[i][1] for i in 1:basis.N]
#     y = [basis.grid[i][2] for i in 1:basis.N]
#     scatter!(p, x, y)

#     display(p)
#     readline()
# end