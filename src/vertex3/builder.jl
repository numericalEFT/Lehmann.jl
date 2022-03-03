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

include("./kernel.jl")
include("./finemesh.jl")

struct GridPoint{D}
    sector::Int                       # sector
    vec::SVector{D,Float}             # actual location of the grid point   
    idx::Int                          # index on the fine meshes
    coord::SVector{D,Int}             # integer coordinate of the grid point on the fine meshes
    rtol::Float                       # the relative error achieved by adding the current grid point 
end

# Base.show(io, grid::GridPoint{D}) =

mutable struct Basis{D}
    ############    fundamental parameters  ##################
    D::Integer  # dimension
    Λ::Float  # UV energy cutoff * inverse temperature
    rtol::Float # error tolerance

    ###############     DLR grids    ###############################
    N::Int # number of basis
    grid::Vector{GridPoint{D}} # grid for the basis
    residual::Vector{Float} # achieved error by each basis

    ###############  linear coefficients for orthognalization #######
    Q::Matrix{Double} # , Q' = R^{-1}, Q*R'= I
    R::Matrix{Double}

    ############ fine grids #################
    mesh::FineMesh{D}

    function Basis{d}(Λ, rtol, projector) where {d}
        _Q = Matrix{Float}(undef, (0, 0))
        _R = similar(_Q)
        return new{d}(d, Λ, rtol, 0, [], [], _Q, _R, FineMesh{d}(Λ, rtol, projector))
    end
end

function addBasis!(basis::Basis{D}, projector, coord, sector = 1) where {D}
    basis.N += 1
    g0 = coord2vec(basis.mesh, coord)
    idx = coord2idx(basis.mesh, coord)
    grid = GridPoint{D}(sector, g0, vec, idx, coord)

    push!(basis.grid, grid)

    basis.Q, basis.R = GramSchmidt(basis, projector)

    # update the residual on the fine mesh
    updateResidual!(basis, projector)
    basis.mesh.selected[idx] = true
    basis.mesh.residual[idx] = 0 # the selected mesh grid has zero residual

    # the new rtol achieved by adding the new grid point
    grid.rtol = sqrt(maximum(mesh.residual))
    return grid
end

function updateResidual!(basis::Basis{D}, projector) where {D}
    Λ, rtol = basis.Λ, basis.rtol
    N, Nfine = basis.N, basis.Nfine

    q = Float.(basis.Q[end, :])
    # q = Double.(basis.Q[end, :])
    fineGrid = basis.fineGrid
    grid = basis.grid
    mesh = basis.mesh

    Threads.@threads for idx in 1:Nfine^D
        # for idx in 1:Nfine^D
        c = idx2coord(mesh, idx)
        if (mesh.selected[idx] == false) && (reducible(D, c) == false)
            g = coord2vec(mesh, c)
            # pp = sum(q[j] * projector(Λ, D, g, grid[j]) for j in 1:N)
            pp = sum(q[j] * projector(Λ, D, g, grid[j].vec, c, grid[j].coord, basis.cache) for j in 1:N)
            _residual = mesh.residual[idx] - pp * pp
            if _residual < 0
                # @warn(c, " grid: ", g, " = ", pp, " and ", _norm, " resudiual: ", Double(_norm)^2 - Double(pp)^2)
                if _residual < -basis.rtol
                    @warn("warning: residual smaller than 0 at $(idx2coord(mesh, idx)) => $g got $(mesh.residual[idx]) - $(pp)^2 = $_residual")
                end
                mesh.residual[idx] = 0
            else
                mesh.residual[idx] = _residual
            end
        end
    end
end

function QR{dim}(Λ, rtol, proj; c0 = nothing, N = nothing) where {dim}
    basis = Basis{dim}(Λ, rtol, proj)
    if isnothing(c0) == false
        for c in c0
            g = addBasis!(basis, proj, c)
            @printf("%3i : ω=(%16.8f, %16.8f; %8i) -> error=%16.6g\n", 1, g.vec[1], g.vec[2], g.sector, g.rtol[end])
        end
    end
    maxResidual, idx = findmax(basis.mesh.residual)

    while isnothing(N) ? sqrt(maxResidual) > rtol : basis.N < N

        c = idx2coord(basis.mesh, idx)

        g = addBasis!(basis, proj, c)
        @printf("%3i : ω=(%16.8f, %16.8f) -> error=%16.8g, Rmin=%16.8g\n", basis.N, g.vec[1], g.vec[2], g.rtol[end], basis.R[end, end])

        for c in mirror(dim, c)
            gp = addBasis!(basis, proj, c)
            @printf("%3i : ω=(%16.8f, %16.8f) -> error=%16.8g, Rmin=%16.8g\n", basis.N, g.vec[1], g.vec[2], g.rtol[end], basis.R[end, end])
        end

        maxResidual, idx = findmax(basis.mesh.residual)

        # plotResidual(basis)
        # testOrthgonal(basis)
    end
    # println(basis.R[:, end])
    # qi = basis.N - 1
    # println("Q", basis.Q[qi, :])
    # overlap = sum(basis.Q[qi, j] * proj(Double(basis.Λ), basis.D, basis.grid[j], basis.grid[end]) for j in 1:basis.N-1)
    # println("overlap: ", overlap, " versus ", basis.R[qi, end])
    # println("QR: ", basis.Q[qi:qi, :] * basis.R')
    # println("QR: ", basis.Q[2:2, :] * basis.R')
    # println(basis.R[:, end-1]' * basis.R[:, end-1])
    # println(proj(basis.Λ, basis.D, basis.grid[end], basis.grid[end]))
    # println("R matrix error:", maximum(abs.(basis.proj - basis.R' * basis.R)))
    testOrthgonal(basis, proj)
    @printf("residual = %.16e\n", maxResidual)
    # plotResidual(basis)
    # plotResidual(basis, proj, Float(0), Float(100), candidate, residual)
    return basis
end

"""
Gram-Schmidt process to the last grid point in basis.grid
"""
function GramSchmidt(basis::Basis{D}, projector) where {D}
    _Q = zeros(Double, (basis.N, basis.N))
    _Q[1:end-1, 1:end-1] = basis.Q

    _R = zeros(Double, (basis.N, basis.N))
    _R[1:end-1, 1:end-1] = basis.R

    _Q[end, end] = 1
    newgrid = basis.grid[end]

    for qi in 1:basis.N-1
        # overlap = sum(_Q[qi, j] * projector(Double(basis.Λ), basis.D, basis.grid[j], basis.grid[end]) for j in 1:basis.N)
        overlap = sum(_Q[qi, j] * projector(basis.Λ, D, basis.grid[j].vec, newgrid.vec) for j in 1:basis.N)
        _Q[end, :] -= overlap * _Q[qi, :]  # <q, qnew> q
        _R[qi, end] = overlap
    end

    # _norm = sqrt(Double(projector(basis.Λ, basis.D, basis.grid[end], basis.grid[end])))
    # for i in 1:basis.N-1
    #     if 1 - (_R[i, end] / _norm)^2 < 0
    #         println(_R[i, end], " and ", _norm)
    #     end
    #     _norm = _norm * sqrt(1 - (_R[i, end] / _norm)^2)
    # end
    # _norm = projector(Double(basis.Λ), basis.D, basis.grid[end], basis.grid[end]) - _R[:, end]' * _R[:, end]
    _norm = projector(basis.Λ, D, basis.grid[end].vec, newgrid.vec) - _R[:, end]' * _R[:, end]
    _norm = sqrt(abs(_norm))
    _R[end, end] = _norm
    _Q[end, :] /= _norm

    if reducible(D, newgrid.coord) == false
        residual = sqrt(basis.mesh.residual[newgrid.idx])
        @assert abs(_norm - residual) < basis.rtol * 100 "inconsistent norm on the grid $(newgrid) $_norm - $residual = $(_norm-residual)"
        if abs(_norm - residual) > basis.rtol * 10
            @warn("inconsistent norm on the grid $(newgrid) $_norm - $residual = $(_norm-residual)")
        end
    end

    return _Q, _R
end

function testOrthgonal(basis::Basis{D}, projector) where {D}
    println("testing orthognalization...")
    KK = zeros(Double, (basis.N, basis.N))
    for i in 1:basis.N
        for j in 1:basis.N
            KK[i, j] = projector(Double(basis.Λ), D, basis.grid[i].vec, basis.grid[j].vec)
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

if abspath(PROGRAM_FILE) == @__FILE__

    Λ = Float(100)
    rtol = Float(1e-8)
    dim = 2
    basis = QR{2}(Float(10), Float(1e-7), projExp_τ)
    @time basis = QR{dim}(Λ, rtol, projExp_τ)

    open("basis.dat", "w") do io
        for i in 1:basis.N
            println(io, basis.grid[i].vec[1], "   ", basis.grid[i].vec[2])
        end
    end
    Nfine = basis.mesh.N
    open("finegrid.dat", "w") do io
        for i in 1:Nfine
            println(io, basis.mesh.fineGrid[i])
        end
    end
    open("residual.dat", "w") do io
        for xi in 1:Nfine
            for yi in 1:Nfine
                if xi <= yi
                    println(io, basis.mesh.residual[coord2idx(mesh, (xi, yi))])
                else
                    println(io, basis.mesh.residual[coord2idx(mesh, (yi, xi))])
                end
            end
        end
    end
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