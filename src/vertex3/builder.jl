using LinearAlgebra:Matrix, zero, similar
using LinearAlgebra, Printf
# using Roots
using Quadmath
# using ProfileView

const Float = Float64
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

# using GR
# function plotResidual(basis)
#     z = Float64.(basis.residualFineGrid)
#     z = reshape(z, (basis.Nfine, basis.Nfine))
#     # contourf(z)
#     p = heatmap(basis.fineGrid, basis.fineGrid, z)
#     # display(p)
#     readline()
# end
# gr()
# data = rand(21, 100)
# heatmap(1:size(data, 1),
#     1:size(data, 2), data,
#     c=cgrad([:blue, :white,:red, :yellow]),
#     xlabel="x values", ylabel="y values",
#     title="My title")

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
    fineGrid::Vector{Float}
    residualFineGrid::Vector{Float}

    function Basis(d, Λ, rtol, projector)
        _Q = Matrix{Float}(undef, (0, 0))
        _grid = Matrix{Float}(undef, (0, 0))
        _finegrid = unilog(Λ, rtol)
        Nfine = length(_finegrid)
        _residualFineGrid = zeros(Float, Nfine^d)

        # initialize the residual on fineGrid with <g, g>
        for (gi, g) in enumerate(iterateFineGrid(d, _finegrid))
            _residualFineGrid[gi] = projector(Λ, dim, g, g)
        end

        # g1 = (_finegrid[1], _finegrid[1])
        # println("compare: ", abs(projector(basis.Λ, basis.D, g1, g1) - _residualFineGrid[1]))
        # for idx in 1:Nfine^d
        #     c = idx2coord(d, Nfine, idx)
        #     # println(c)
        #     g = (_finegrid[c[1]], _finegrid[c[2]])
        #     _residualFineGrid[idx] = projector(Λ, d, g, g)
        # end
        # println(_residualFineGrid)

        return new(d, Λ, rtol, 0, _grid, [], _Q, similar(_Q), Nfine, _finegrid, _residualFineGrid)
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

function idx2coord(dim, N, idx)
    return ((idx - 1) ÷ N + 1, (idx - 1) % N + 1)
end

function coord2idx(dim, N, coord)
    return (coord[1] - 1) * N + coord[2]
end

"""
composite expoential grid
"""
function unilog(Λ, rtol)
    # N = 500
    # grid = [i / N * Λ for i in 1:N]
    # return grid

    degree = 8
    ratio = Float(1.2)
    N = Int(floor(log(Λ) / log(ratio) + 1))
    panel = [Λ / ratio^(N - i) for i in 1:N]
    grid = Vector{Float}(undef, 0)
    for i in 1:length(panel) - 1
        uniform = [panel[i] + (panel[i + 1] - panel[i]) / degree * j for j in 0:degree - 1]
        append!(grid, uniform)
    end
    append!(grid, Λ)
    println("Composite expoential grid size: $(length(grid))")
    return grid
end

function addBasis!(basis, projector, g0)
    basis.N += 1

    _grid = copy(basis.grid)
    _Q = copy(basis.Q)
    basis.grid = zeros(Float, (basis.N, basis.D))
    basis.Q = zeros(Float, (basis.N, basis.N))

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
        basis.Q[end, :] = mGramSchmidt(basis, g0)
    end

    updateResidual!(basis, projector)
    append!(basis.residual, sqrt(maximum(basis.residualFineGrid))) # record error after the new grid is added

end

function updateResidual!(basis, projector)
    q = basis.Q[end, :]
    # for (idx, g) in enumerate(iterateFineGrid(basis.D, basis.fineGrid))
    for idx in 1:basis.Nfine^basis.D
        c = idx2coord(basis.D, basis.Nfine, idx)
        if c[1] <= c[2]
            g = (basis.fineGrid[c[1]], basis.fineGrid[c[2]])
            proj = Float(0)
            for j in 1:basis.N
                proj += q[j] * projector(basis.Λ, basis.D, g, basis.grid[j, :])
            end

            basis.residualFineGrid[idx] -= proj^2
            if basis.residualFineGrid[idx] < Float(0)
                if basis.residualFineGrid[idx] < Float(-basis.rtol / 1000)
                    println("warning: residual smaller than 0 at $(idx2coord(basis.D, basis.Nfine, idx)) has $(basis.residualFineGrid[idx])")
                end
                basis.residualFineGrid[idx] = Float(0)
            end

            ############  Mirror symmetry  #############################
            idxp = coord2idx(basis.D, basis.Nfine, (c[2], c[1]))
            basis.residualFineGrid[idxp] = basis.residualFineGrid[idx]
        end
    end
end

function QR(dim, Λ, rtol, proj; g0=nothing, N=nothing)
    basis = Basis(dim, Λ, rtol, proj)
    if isnothing(g0) == false
        for g in g0
            addBasis!(basis, proj, g)
            @printf("%3i : ω=(%24.8f, %24.8f) -> error=%24.16g\n", 1, g[1], g[2], basis.residual[end])
        end
    end
    maxResidual, idx = findmax(basis.residualFineGrid)

    while isnothing(N) ? sqrt(maxResidual) > rtol / 10 : basis.N < N

        c = idx2coord(basis.D, basis.Nfine, idx)
        g = (basis.fineGrid[c[1]], basis.fineGrid[c[2]])
        residual = Residual(basis, proj, g)

        addBasis!(basis, proj, g)
        @printf("%3i : ω=(%24.8f, %24.8f) -> error=%24.16g\n", basis.N, g[1], g[2], basis.residual[end])

        if abs(g[1] - g[2]) > 1e-8
            gp = (g[2], g[1])
            addBasis!(basis, proj, gp)
            @printf("%3i : ω=(%24.8f, %24.8f) -> error=%24.16g\n", basis.N, gp[1], gp[2], basis.residual[end])
        end

        maxResidual, idx = findmax(basis.residualFineGrid)

        # plotResidual(basis)
    end
    testOrthgonal(basis)
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
projqq(basis, q1::Vector{Float}, q2::Vector{Float}) = q1' * basis.proj * q2

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

function Residual(basis, proj, g)
    # norm2 = proj(g, g) - \sum_i (<qi, K_g>)^2
    # qi=\sum_j Q_ij K_j ==> (<qi, K_g>)^2 = (\sum_j Q_ij <K_j, K_g>)^2 = \sum_jk Q_ij*Q_ik <K_j, K_g>*<K_k, Kg>
    
    KK = [proj(basis.Λ, basis.D, basis.grid[j, :], g) for j in 1:basis.N]
    norm2 = proj(basis.Λ, basis.D, g, g) - (norm(basis.Q * KK))^2
    # norm2 = proj(basis.Λ, basis.D, g, g)
    # for i in 1:basis.N
    #     norm2 -= (sum(basis.Q[i, :] .* KK))^2
    # end
    return norm2 < 0 ? Float(0) : sqrt(norm2) 
    # return norm2
end
        
function testOrthgonal(basis)
    println("testing orthognalization...")
    II = basis.Q * basis.proj * basis.Q'
    maxerr = maximum(abs.(II - I))
println("Max Orthognalization Error: ", maxerr)
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
    @time ωBasis = QR(dim, Λ, rtol, projExp_τ)
    # @time τBasis = QR(Λ / 2, 1e-11, projPHA_τ, Float(0), N=ωBasis.N)
    # nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :acorr)

    # @time basis = QR(100, 1e-10)
    # readline()
    # basis = QR(100, 1e-3)

end