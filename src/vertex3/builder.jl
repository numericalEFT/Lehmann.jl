using Printf:Threads
using Quadmath:BigFloat
using LinearAlgebra:Matrix, zero, similar, Threads
using LinearAlgebra, Printf
# using Roots
using Quadmath
# using ProfileView
using InteractiveUtils

const Float = Float128
const FloatL = Float64
# const Float = BigFloat
# const Float = Float128

include("./kernel.jl")


# using Plots
# function plotResidual(basis, proj, gmin, gmax, candidate=nothing, residual=nothing)
#     ω = LinRange(gmin, gmax, 1000)
#     y = [Residual(basis, proj, w) for w in ω]
#     p = plot(ω, y, xlims=(gmin, gmax))
#     if isnothing(candidate) == false
#         plot!(p, candidate, residual, seriestype=:scatter)
#     end
#     display(p)
#     readline()
# end

using Plots
gr()

function plotResidual(basis)
    z = Float64.(basis.residualFineGrid)
    z = reshape(z, (basis.Nfine, basis.Nfine))
    # contourf(z)
    p = heatmap(basis.fineGrid, basis.fineGrid, z)
    # p = heatmap(z)
    x = [basis.grid[i, 1] for i in 1:basis.N]
    y = [basis.grid[i, 2] for i in 1:basis.N]
    scatter!(p, x, y)

    display(p)
    readline()
end

mutable struct Basis
    ############ fundamental parameters  ##################
    D::Integer  # dimension
    Λ::Float  # UV energy cutoff * inverse temperature
    rtol::Float # error tolerance

    ########### DLR grids    ###############################
    N::Int # number of basis
    grid::Matrix{Float} # grid for the basis
    residual::Vector{Float} # achieved error by each basis
    Q::Matrix{Float} # K = Q*R
    proj::Matrix{Float} # the overlap of basis functions <K(g_i), K(g_j)>

    ##### fine grids and the their residuals #################
    Nfine::Integer
    fineGrid::Vector{FloatL}
    residualFineGrid::Vector{FloatL}
    gridIdx::Vector{Int} # grid for the basis

    function Basis(d, Λ, rtol, projector)
        _Q = Matrix{Float}(undef, (0, 0))
        _grid = Matrix{Float}(undef, (0, 0))

        # initialize the residual on fineGrid with <g, g>
        _finegrid = FloatL.(unilog(Λ, rtol))
        Nfine = length(_finegrid)
        _residualFineGrid = zeros(Float, Nfine^d)
        for (gi, g) in enumerate(iterateFineGrid(d, _finegrid))
            _residualFineGrid[gi] = projector(Λ, dim, g, g)
        end

        return new(d, Λ, rtol, 0, _grid, [], _Q, similar(_Q), Nfine, _finegrid, _residualFineGrid, [])
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
    return (((idx - 1) ÷ N + 1, (idx - 1) % N + 1))
end

function coord2idx(dim::Int, N::Int, coord)
    return Int((coord[1] - 1) * N + coord[2])
end

"""
composite expoential grid
"""
function unilog(Λ, rtol)
    # N = 500
    # grid = [i / N * Λ for i in 1:N]
    # return grid

    degree = 8
    ratio = FloatL(1.4)
    N = Int(floor(log(Λ) / log(ratio) + 1))
    panel = [Λ / ratio^(N - i) for i in 1:N]
    grid = Vector{FloatL}(undef, 0)
    for i in 1:length(panel) - 1
        uniform = [panel[i] + (panel[i + 1] - panel[i]) / degree * j for j in 0:degree - 1]
        append!(grid, uniform)
    end
    append!(grid, Λ)
    println("Composite expoential grid size: $(length(grid))")
    return grid

    # filename = string(@__DIR__, "/../functional/basis/$(string(:corr))/dlr$(Int(Λ))_1e$(1e-6).dlr")

    # if isfile(filename) == false
    #     error("$filename doesn't exist. DLR basis hasn't been generated.")
    # end

    # grid = readdlm(filename)
    # # println("reading $filename")

    # ω = grid[:, 2]

    # degree = 8
    # ratio = Float(1.4)
    # N = Int(floor(log(Λ) / log(ratio) + 1))
    # panel = [Λ / ratio^(N - i) for i in 1:N]
    # grid = Vector{Float}(undef, 0)
    # for i in 1:length(panel) - 1
    #     uniform = [panel[i] + (panel[i + 1] - panel[i]) / degree * j for j in 0:degree - 1]
    #     append!(grid, uniform)
    # end
    # append!(grid, Λ)
    # println("Composite expoential grid size: $(length(grid))")
    # return grid
end

function addBasis!(basis, projector, coord)
    basis.N += 1
    g0 = Float.((basis.fineGrid[coord[1]], basis.fineGrid[coord[2]]))
    # g0 = Float.(g0)

    _grid = copy(basis.grid)
    _Q = copy(basis.Q)
    _grididx = copy(basis.gridIdx)

    basis.grid = zeros(Float, (basis.N, basis.D))
    basis.Q = zeros(Float, (basis.N, basis.N))

    # basis.grid[1:end - 1, :] = _grid
    # basis.grid[end, :] .= g0

    # basis.gridIdx[1:end - 1, :] .= _grididx
    # basis.gridIdx[end, :] .= coord

    # basis.proj = projKernel(basis, projector)

    if basis.N == 1
        idx = 1
        basis.grid[1, :] .= g0
        basis.proj = projKernel(basis, projector)
        basis.Q[1,1] = 1 / sqrt(projector(basis.Λ, basis.D, g0, g0))
    else
        basis.grid[1:end - 1, :] = _grid
        basis.grid[end, :] .= g0
        basis.proj = projKernel(basis, projector)
        basis.Q[1:end - 1, 1:end - 1] = _Q
        basis.Q[end, :] = GramSchmidt(basis, g0)
    end

    # @code_warntype updateResidual!(basis, projector)
    updateResidual!(basis, projector)
    append!(basis.residual, sqrt(maximum(basis.residualFineGrid))) # record error after the new grid is added
    append!(basis.gridIdx, coord2idx(basis.D, basis.Nfine, coord))
    return g0
end

function updateResidual!(basis, projector)
    q = FloatL.(basis.Q[end, :])
    Λ = FloatL(basis.Λ)
    rtol = FloatL(basis.rtol)
    fineGrid = FloatL.(basis.fineGrid)
    grid = FloatL.(basis.grid)
    N::Int, Nfine::Int, D::Int = basis.N, basis.Nfine, basis.D

    Threads.@threads for idx in 1:Nfine^D
        # println(Threads.threadid())
        c = idx2coord(D, Nfine, idx)
        if c[1] <= c[2]
            g = (fineGrid[c[1]], fineGrid[c[2]])
            KK = [projector(Λ, D, g, grid[j, :]) for j in 1:N]
            basis.residualFineGrid[idx] -= (q' * KK)^2

            # proj = FloatL(0)
            # for j in 1:N
            #     proj += q[j] * projector(Λ, D, g, grid[j, :])
            # end

            # basis.residualFineGrid[idx] -= proj^2

            # if basis.residualFineGrid[idx] < FloatL(0)
            #     if basis.residualFineGrid[idx] < FloatL(-rtol / 1000)
            #         println("warning: residual smaller than 0 at $(idx2coord(D, Nfine, idx)) has $(basis.residualFineGrid[idx])")
            #         exit(0)
            #     end
            #     basis.residualFineGrid[idx] = FloatL(0)
            # end

        end
    end

    ############  Mirror symmetry  #############################
    for idx in 1:Nfine^D
        # println(Threads.threadid())
        c = idx2coord(D, Nfine, idx)
        if c[1] > c[2]
            idxp = coord2idx(basis.D, basis.Nfine, (c[2], c[1]))
            basis.residualFineGrid[idx] = basis.residualFineGrid[idxp]
        end
    end
end

function QR(dim, Λ, rtol, proj; c0=nothing, N=nothing)
    basis = Basis(dim, Λ, rtol, proj)
    if isnothing(c0) == false
        for c in c0
            g = addBasis!(basis, proj, c)
            @printf("%3i : ω=(%24.8f, %24.8f) -> error=%24.16g\n", 1, g[1], g[2], basis.residual[end])
        end
    end
    maxResidual, idx = findmax(basis.residualFineGrid)

    while isnothing(N) ? sqrt(maxResidual) > rtol : basis.N < N

        c = idx2coord(basis.D, basis.Nfine, idx)

        # residual = Residual(basis, proj, g)
        # residualp = Residual(basis, proj, g, 5)
        # println("$c has diff: ", abs(residual - residualp) / residual, "  for $residual vs $residualp")

        g = addBasis!(basis, proj, c)
        @printf("%3i : ω=(%24.8f, %24.8f) -> error=%24.16g\n", basis.N, g[1], g[2], basis.residual[end])
        # testOrthgonal(basis)

        if c[1] != c[2]
            gp = addBasis!(basis, proj, (c[2], c[1]))
            @printf("%3i : ω=(%24.8f, %24.8f) -> error=%24.16g\n", basis.N, gp[1], gp[2], basis.residual[end])
            # testOrthgonal(basis)
        end

        maxResidual, idx = findmax(basis.residualFineGrid)

        # plotResidual(basis)
    end
    testOrthgonal(basis)
    testResidual(basis, proj)
    @printf("residual = %.16e\n", sqrt(maxResidual))
    plotResidual(basis)
# plotResidual(basis, proj, Float(0), Float(100), candidate, residual)
    return basis
end

"""
q1=sum_j c_j K_j
q2=sum_k d_k K_k
return <q1, q2> = sum_jk c_j*d_k <K_j, K_k>
"""
# projqq(basis, q1::Vector{Float}, q2::Vector{Float}) = q1' * basis.proj * q2
# projqq(basis, q1, q2) = BigFloat.(q1)' * BigFloat.(basis.proj) * BigFloat.(q2)
projqq(basis, q1, q2) = BigFloat.(q1)' * BigFloat.(basis.proj) * BigFloat.(q2)

"""
<K(g_i), K(g_j)>
"""    
function projKernel(basis, proj)
    K = zeros(Float, (basis.N, basis.N))
    for i in 1:basis.N
        for j in 1:basis.N
            K[i,j] = proj(basis.Λ, basis.D, basis.grid[i, :], basis.grid[j, :])
        end
    end
    return K
end
    
"""
modified Gram-Schmidt process
"""
    function mGramSchmidt(basis, g)
    qnew = zeros(Float, basis.N)
    qnew[end] = 1
    # println(basis.proj)
        
    for qi in 1:basis.N - 1
        q = basis.Q[qi, :]
        qnew -= projqq(basis, q, qnew) .* q  # <q, qnew> q
        # println("q: ", q)
        # println("proj:", projqq(basis, q, qnew))
    end
    # println(qnew)
    return qnew / sqrt(abs(projqq(basis, qnew, qnew)))
end

"""
Gram-Schmidt process
"""
function GramSchmidt(basis, g)
    qnew = zeros(Float, basis.N)
    qnew[end] = 1
    q0 = copy(qnew)
    # println(basis.proj)
        
    for qi in 1:basis.N - 1
        q = basis.Q[qi, :]
        qnew -= projqq(basis, q, q0) .* q  # <q, qnew> q
    end

    # q0 = copy(qnew    # println("GS: ", projqq(basis, qnew, qnew))
    # println(qnew)
    # end
    # println("GS: ", projqq(basis, qnew, qnew))
    # println(qnew)
    qnorm = qnew / sqrt(abs(projqq(basis, qnew, qnew)))
    # println(Float64.(qnorm)' * basis.proj * Float64.(qnorm) - 1)
    return qnorm
end

# """
# Gram-Schmidt process
# """
# function GramSchmidt(basis, idx, g::Float)
#     q0 = zeros(Float, basis.N)
#     q0[idx] = 1
#     qnew = copy(q0)
        
#     for qi in 1:basis.N
#         if qi == idx
#     continue
#     end
#         q = basis.Q[qi, :]
#         qnew -=  projqq(basis, q, q0) .* q
#     end
    
#     norm = sqrt(projqq(basis, qnew, qnew))
#     return qnew / norm
# end

within(g, g0, cutoff) = g0[1] < 5 || g0[2] < 5 || g[1] < 5 || g[2] < 5 || ((g[1] / cutoff <= g0[1] <= g[1] * cutoff) && (g[2] / cutoff <= g0[2] <= g[2] * cutoff))

function Residual(basis, proj, g, cutoff=-1)
    # norm2 = proj(g, g) - \sum_i (<qi, K_g>)^2
    # qi=\sum_j Q_ij K_j ==> (<qi, K_g>)^2 = (\sum_j Q_ij <K_j, K_g>)^2 = \sum_jk Q_ij*Q_ik <K_j, K_g>*<K_k, Kg>
    
    if cutoff < 0
        KK = [proj(basis.Λ, basis.D, basis.grid[j, :], g) for j in 1:basis.N]
        norm2 = proj(basis.Λ, basis.D, g, g) - (norm(basis.Q * KK))^2
        return norm2
    else
        KK = [proj(basis.Λ, basis.D, basis.grid[j, :], g) for j in 1:basis.N if within(basis.grid[j, :], g, cutoff)]
        norm2 = proj(basis.Λ, basis.D, g, g)
        for i in 1:basis.N
            if within(basis.grid[i, :], g, cutoff)
                pr = Float(0)
                for j in 1:basis.N 
                    gp = basis.grid[j, :]
                    pr += basis.Q[i, j] * proj(basis.Λ, basis.D, g, gp)
                end
                norm2 -= pr^2
            end
        end
        return norm2
    end
    # norm2 = proj(basis.Λ, basis.D, g, g)
    # for i in 1:basis.N
    #     norm2 -= (sum(basis.Q[i, :] .* KK))^2
    # end
    # return norm2 < 0 ? Float(0) : sqrt(norm2) 
    # return norm2
end
        
function testOrthgonal(basis)
    println("testing orthognalization...")
    II = basis.Q * basis.proj * basis.Q'
    maxerr = maximum(abs.(II - I))
println("Max Orthognalization Error: ", maxerr)
# display(II - I)
# println()
end

function testResidual(basis, proj)
    # residual = [Residual(basis, proj, basis.grid[i, :]) for i in 1:basis.N]
    # println("Max deviation from zero residual: ", maximum(abs.(residual)))
    println("Max deviation from zero residual on the DLR grids: ", maximum(abs.(basis.residualFineGrid[basis.gridIdx])))
end
    
"""
#Arguments:
- `type`: type of kernel, :fermi, :boson
- `Λ`: cutoff = UV Energy scale of the spectral density * inverse temperature
- `rtol`: tolerance absolute error
"""
function dlr_functional(type, Λ, rtol)
    Λ = Float(Λ)
    println("Building ω grid ... ")
    ωBasis = QR(2, Λ, rtol, projPH_ω, Λ)
    # println("Building τ grid ... ")
    # τBasis = tauGrid(ωBasis.grid, ωBasis.N, Λ, rtol, :corr)
    # # τBasis = QR(Λ / 2, rtol / 10, projPH_τ, Float(0), N=ωBasis.N)
    # println("Building n grid ... ")
    # nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :corr)
    
    rank = ωBasis.N
    ωGrid = ωBasis.grid
    # τGrid = τBasis / Λ
    τGrid = τBasis
    nGrid = nBasis
    ########### output  ############################
    @printf("%5s  %32s  %32s  %8s\n", "index", "real freq", "tau", "ωn")
        for r in 1:rank
        @printf("%5i  %32.17g  %32.17g  %16i\n", r, ωGrid[r], τGrid[r], nGrid[r])
    end

    dlr = Dict([(:ω, ωGrid), (:τ, τGrid), (:ωn, nGrid)])
end

if abspath(PROGRAM_FILE) == @__FILE__    
    # freq, Q = findBasis(1.0e-3, Float(100))
    # basis = QR(100, 1e-3)
    Λ = Float(100)
    rtol = Float(1e-6)
    dim = 2
    # g0 = [[Float(0), Float(0)], [Float(0), Λ], [Λ, Float(0)], [Λ, Λ]]
    # g0 = [[Λ, Λ], [Float(0), Λ], [Λ, Float(0)]]
    # println(unilog(Λ, rtol))
    # Λ = 100
    # @time ωBasis = QR(dim, Λ, rtol, projExp_τ, N=100)
    # @time ωBasis = QR(dim, Λ, rtol / 10, projExp_τ)
    @time ωBasis = QR(dim, Λ, rtol / 10, projExp_τ)
    # @code_warntype QR(dim, Λ, rtol / 10, projExp_τ)
    # @time τBasis = QR(Λ / 2, 1e-11, projPHA_τ, Float(0), N=ωBasis.N)
    # nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :acorr)

    # @time basis = QR(100, 1e-10)
    # readline()
    # basis = QR(100, 1e-3)

end