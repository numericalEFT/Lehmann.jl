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

mutable struct Basis{D}
    ############    fundamental parameters  ##################
    D::Integer  # dimension
    Λ::Float  # UV energy cutoff * inverse temperature
    rtol::Float # error tolerance

    ###############     DLR grids    ###############################
    N::Int # number of basis
    grid::Vector{SVector{D,Float}} # grid for the basis
    residual::Vector{Float} # achieved error by each basis
    # Q::Matrix{Float} # K = Q*R
    Q::Matrix{Double} # , Q' = R^{-1}, Q*R'= I
    R::Matrix{Double}

    ############ fine grids #################
    Nfine::Integer
    fineGrid::Vector{Float}
    cache::Vector{Float}

    ########## residual defined on the fine grids #################
    residualFineGrid::Vector{Double} #length = Nfine^D/D!
    selectedFineGrid::Vector{Bool}
    gridIdx::Vector{Int} # grid for the basis
    gridCoord::Vector{SVector{D,Int}}
    gridProj::Vector{Vector{Float}}

    function Basis{d}(Λ, rtol, projector) where {d}
        _Q = Matrix{Float}(undef, (0, 0))

        # initialize the residual on fineGrid with <g, g>
        _finegrid = Float.(unilog(Λ, rtol))
        Nfine = length(_finegrid)
        _cache = zeros(Float, Nfine)
        for (gi, g) in enumerate(_finegrid)
            _cache[gi] = exp(-g)
        end

        _residualFineGrid = zeros(Float, Nfine^d)
        _selectedFineGrid = zeros(Bool, Nfine^d)
        for (gi, g) in enumerate(iterateFineGrid(d, _finegrid))
            c1, c2 = idx2coord(d, Nfine, gi)
            if c1 <= c2
                # _residualFineGrid[gi] = sqrt(projector(Double(Λ), d, g, g))
                _residualFineGrid[gi] = projector(Λ, d, g, g)
            end
        end
        return new{d}(d, Λ, rtol, 0, [], [], _Q, similar(_Q), Nfine, _finegrid, _cache,
            _residualFineGrid, _selectedFineGrid, [], [], [])
    end
end

function iterateFineGrid(dim, _finegrid)
    if dim == 1
        return _finegrid
    elseif dim == 2
        return Iterators.product(_finegrid, _finegrid)
    else # d==3
        return Iterators.product(_finegrid, _finegrid, _finegrid)
    end
end

function idx2coord(dim::Int, N::Int, idx::Int)
    return (((idx - 1) % N + 1, (idx - 1) ÷ N + 1))
end

function coord2idx(dim::Int, N::Int, coord)
    return Int((coord[2] - 1) * N + coord[1])
end

"""
composite expoential grid
"""
function unilog(Λ, rtol)
    ############## use composite grid #############################################
    # degree = 8
    # ratio = Float(1.4)
    # N = Int(floor(log(Λ) / log(ratio) + 1))
    # panel = [Λ / ratio^(N - i) for i in 1:N]
    # grid = Vector{Float}(undef, 0)
    # for i in 1:length(panel)-1
    #     uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
    #     append!(grid, uniform)
    # end
    # append!(grid, Λ)
    # println(grid)
    # println("Composite expoential grid size: $(length(grid))")
    # return grid

    ############# DLR based fine grid ##########################################
    dlr = DLRGrid(Euv = Float64(Λ), beta = 1.0, rtol = Float64(rtol) / 100, isFermi = true, symmetry = :ph, rebuild = true)
    # println("fine basis number: $(dlr.size)\n", dlr.ω)
    degree = 4
    grid = Vector{Float}(undef, 0)
    panel = Float.(dlr.ω)
    for i in 1:length(panel)-1
        uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
        append!(grid, uniform)
    end
    println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[2])]")
    return grid
end

function addBasis!(basis::Basis{D}, projector, coord) where {D}
    basis.N += 1
    g0 = SVector{D,Float}([basis.fineGrid[coord[1]], basis.fineGrid[coord[2]]])
    idx = coord2idx(basis.D, basis.Nfine, coord)

    push!(basis.grid, g0)
    push!(basis.gridIdx, idx)
    push!(basis.gridCoord, coord)
    # println(coord, "->", g0, " -> ", basis.gridIdx[end])

    GramSchmidt!(basis, projector)

    updateResidual!(basis, projector)
    basis.selectedFineGrid[idx] = true
    basis.residualFineGrid[idx] = 0
    push!(basis.residual, sqrt(maximum(basis.residualFineGrid))) # record error after the new grid is added
    return g0
end

function updateResidual!(basis::Basis{D}, projector) where {D}
    Λ, rtol = basis.Λ, basis.rtol
    N, Nfine = basis.N, basis.Nfine

    # q = Float.(basis.Q[end, :])
    q = Double.(basis.Q[end, :])
    fineGrid = basis.fineGrid
    grid = basis.grid

    # Threads.@threads for idx in 1:Nfine^D
    for idx in 1:Nfine^D
        c = idx2coord(D, Nfine, idx)
        if c[1] <= c[2] && (basis.selectedFineGrid[idx] == false)
            # if (basis.selectedFineGrid[idx] == false)
            g = (fineGrid[c[1]], fineGrid[c[2]])
            # pp = sum(q[j] * projector(Λ, D, g, grid[j]) for j in 1:N)
            pp = sum(q[j] * projector(Λ, D, g, grid[j], c, basis.gridCoord[j], basis.cache) for j in 1:N)
            _residual = basis.residualFineGrid[idx]
            # println("pp", pp)
            # println("norm ", _norm)

            # if abs(pp) > abs(_norm) || abs(_norm) < eps(Float(1)) * 10
            # if abs(pp) > abs(_norm)
            if _residual - pp^2 < 0
                # @warn(c, " grid: ", g, " = ", pp, " and ", _norm, " resudiual: ", Double(_norm)^2 - Double(pp)^2)
                if _residual - pp^2 < -basis.rtol
                    @warn("warning: residual smaller than 0 at $(idx2coord(D, Nfine, idx)) => $g got $_residual - $(pp)^2 = $(_residual-pp^2)")
                end
                basis.residualFineGrid[idx] = 0
            else
                # basis.residualFineGrid[idx] = _norm * sqrt((1 - pp / _norm) * (1 + pp / _norm))
                # basis.residualFineGrid[idx] = sqrt(Double(_norm)^2 - Double(pp)^2)
                basis.residualFineGrid[idx] = _residual - pp^2
            end
        end
    end
end

function QR{dim}(Λ, rtol, proj; c0 = nothing, N = nothing) where {dim}
    basis = Basis{dim}(Λ, rtol, proj)
    if isnothing(c0) == false
        for c in c0
            g = addBasis!(basis, proj, c)
            @printf("%3i : ω=(%16.8f, %16.8f) -> error=%16.6g\n", 1, g[1], g[2], basis.residual[end])
        end
    end
    maxResidual, idx = findmax(basis.residualFineGrid)

    while isnothing(N) ? sqrt(maxResidual) > rtol : basis.N < N

        c = idx2coord(basis.D, basis.Nfine, idx)

        g = addBasis!(basis, proj, c)
        @printf("%3i : ω=(%16.8f, %16.8f) -> error=%16.8g, Rmin=%16.8g\n", basis.N, g[1], g[2], basis.residual[end], basis.R[end, end])

        if c[1] != c[2]
            gp = addBasis!(basis, proj, (c[2], c[1]))
            @printf("%3i : ω=(%16.8f, %16.8f) -> error=%16.8g, Rmin=%16.8g\n", basis.N, gp[1], gp[2], basis.residual[end], basis.R[end, end])
        end

        maxResidual, idx = findmax(basis.residualFineGrid)

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
    testOrthgonal(basis)
    testResidual(basis, proj)
    @printf("residual = %.16e\n", maxResidual)
    # plotResidual(basis)
    # plotResidual(basis, proj, Float(0), Float(100), candidate, residual)
    return basis
end

"""
q1=sum_j c_j K_j
q2=sum_k d_k K_k
return <q1, q2> = sum_jk c_j*d_k <K_j, K_k>
"""
# projqq(_proj, q1, q2) = q1' * _proj * q2
# function projqq(basis, q1, q2)
#     return dot(basis.L' * q1, basis.U * q2)
# end

"""
add the last grid point to the overlap matrix: 
basis.proj = R'*R = <K(g_i), K(g_j)>
"""
# function overlap(basis, projector)
#     _proj = zeros(Float, (basis.N, basis.N))
#     _proj[1:end-1, 1:end-1] = basis.proj
#     for i in 1:basis.N
#         p = projector(FloatL(basis.Λ), basis.D, basis.grid[end], basis.grid[i])
#         # pp = proj(basis.Λ, basis.D, basis.grid[end], basis.grid[i], coord[end], coord[i], basis.cache)
#         # @assert abs(p - pp) < 1e-16 "$p vs $pp"
#         _proj[end, i] = p
#         _proj[i, end] = p
#     end
#     return _proj
# end

"""
Gram-Schmidt process to the last grid point in basis.grid
"""
function GramSchmidt!(basis, projector)
    # _Q = zeros(Float, (basis.N, basis.N))
    _Q = zeros(Double, (basis.N, basis.N))
    _Q[1:end-1, 1:end-1] = basis.Q

    _R = zeros(Double, (basis.N, basis.N))
    _R[1:end-1, 1:end-1] = basis.R

    _Q[end, end] = 1
    # qnew = zeros(Double, basis.N)
    # qnew[end] = 1

    # A = cholesky(basis.proj)

    for qi in 1:basis.N-1
        # overlap = sum(_Q[qi, j] * projector(Double(basis.Λ), basis.D, basis.grid[j], basis.grid[end]) for j in 1:basis.N)
        overlap = sum(_Q[qi, j] * projector(basis.Λ, basis.D, basis.grid[j], basis.grid[end]) for j in 1:basis.N)
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
    _norm = projector(basis.Λ, basis.D, basis.grid[end], basis.grid[end]) - _R[:, end]' * _R[:, end]
    # @assert _norm > 0.0 "$_norm is negative on the grid $(basis.grid[end])"
    # @assert _norm < -eps(Double(1)) * 10 "$_norm is negative on the grid $(basis.grid[end])"
    _norm = sqrt(abs(_norm))
    _R[end, end] = _norm
    _Q[end, :] /= _norm

    c = basis.gridCoord[end]
    if c[1] <= c[2]
        residual = sqrt(basis.residualFineGrid[basis.gridIdx[end]])
        @assert abs(_norm - residual) < basis.rtol * 100 "inconsistent norm on the grid $(basis.grid[end]) $_norm - $residual = $(_norm-residual)"
        if abs(_norm - residual) > basis.rtol * 10
            @warn("inconsistent norm on the grid $(basis.grid[end]) $_norm - $residual = $(_norm-residual)")
        end
    end

    basis.Q = _Q
    basis.R = _R
end

function Residual(basis, proj, g)
    # norm2 = proj(g, g) - \sum_i (<qi, K_g>)^2
    # qi=\sum_j Q_ij K_j ==> (<qi, K_g>)^2 = (\sum_j Q_ij <K_j, K_g>)^2 = \sum_jk Q_ij*Q_ik <K_j, K_g>*<K_k, Kg>

    KK = [proj(basis.Λ, basis.D, basis.grid[j], g) for j in 1:basis.N]
    norm2 = proj(basis.Λ, basis.D, g, g) - (norm(basis.Q * KK))^2
    return norm2
end

function testOrthgonal(basis)
    println("testing orthognalization...")
    # A = cholesky(basis.proj)
    # QL = basis.Q * A.L
    # UQ = A.U * basis.Q'
    # II = QL * UQ
    RQ = Double.(basis.R) * Double.(basis.Q')
    II = RQ' * RQ
    # println(diag(II) .- 1)
    maxerr = maximum(abs.(II - I))
    println("Max Orthognalization Error: ", maxerr)
    # display(II - I)
    # println()
    # for i = 1:basis.N
    #     println(i, " -> ", dot(A.U * basis.Q[i, :], A.U * basis.Q[i, :]) - 1)
    # end
end

function testResidual(basis, proj)
    # residual = [Residual(basis, proj, basis.grid[i, :]) for i in 1:basis.N]
    # println("Max deviation from zero residual: ", maximum(abs.(residual)))
    println("Max deviation from zero residual on the DLR grids: ", maximum(abs.(basis.residualFineGrid[basis.gridIdx])))
end

if abspath(PROGRAM_FILE) == @__FILE__

    Λ = Float(1000)
    rtol = Float(1e-9)
    dim = 2
    basis = QR{2}(Float(10), Float(1e-7), projExp_τ)
    @time basis = QR{dim}(Λ, rtol, projExp_τ)

    open("basis.dat", "w") do io
        for i in 1:basis.N
            println(io, basis.grid[i][1], "   ", basis.grid[i][2])
        end
    end
    open("finegrid.dat", "w") do io
        for i in 1:basis.Nfine
            println(io, basis.fineGrid[i])
        end
    end
    open("residual.dat", "w") do io
        for xi in 1:basis.Nfine
            for yi in 1:basis.Nfine
                if xi <= yi
                    println(io, basis.residualFineGrid[coord2idx(2, basis.Nfine, (xi, yi))])
                else
                    println(io, basis.residualFineGrid[coord2idx(2, basis.Nfine, (yi, xi))])
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