using LinearAlgebra:Matrix, zero, similar
using LinearAlgebra, Printf
# using Roots
using Quadmath
# using ProfileView

# const Float = Float64
# const Float = BigFloat
const Float = Float128

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

using GR
function plotResidual(basis)
    z = Float64.(basis.residualFineGrid)
    z = reshape(z, (basis.Nfine, basis.Nfine))
    # contourf(z)
    p = heatmap(z)
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
        _residualFineGrid = zeros(Nfine^d)

        # initialize the residual on fineGrid with <g, g>
        for (gi, g) in enumerate(iterateFineGrid(d, _finegrid))
            _residualFineGrid[gi] = projector(Λ, dim, g, g)
        end
        println(_residualFineGrid)

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

"""
composite expoential grid
"""
function unilog(Λ, rtol)
    degree = 2
    ratio = Float(1.5)
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
        basis.grid[1, :] = g0
        basis.proj = projKernel(basis, projector)
        basis.Q[1,1] = 1 / sqrt(projector(basis.Λ, basis.D, g0, g0))
    else
        basis.grid[1:end - 1, :] = _grid
        basis.grid[end, :] = g0
        basis.proj = projKernel(basis, projector)
        basis.Q[1:end - 1, 1:end - 1] = _Q
        basis.Q[end, :] = mGramSchmidt(basis, g0)
    end

    updateResidual!(basis, projector)
    append!(basis.residual, maximum(basis.residualFineGrid)) # record error after the new grid is added

end

function updateResidual!(basis, projector)
    q = basis.Q[end, :]
    for (idx, g) in enumerate(iterateFineGrid(basis.D, basis.fineGrid))
        proj = Float(0)
        for j in 1:basis.N
            proj += q[j] * projector(basis.Λ, basis.D, g, basis.grid[j, :])
        end
        # return norm2 < 0 ? Float(0) : sqrt(norm2) 
        basis.residualFineGrid[idx] -= proj^2
    end
    # println(basis.residualFineGrid)
end

function QR(dim, Λ, rtol, proj, g0; N=nothing)
    basis = Basis(dim, Λ, rtol, proj)
    for g in g0
        addBasis!(basis, proj, g)
        @printf("%3i : ω=(%24.8f, %24.8f) -> error=%24.16g\n", 1, g[1], g[2], basis.residual[end])
        plotResidual(basis)
    end
    maxResidual, ωi = findmax(residual)

    while isnothing(N) ? maxResidual > rtol / 10 : basis.N < N

        g = candidate[ωi]
        addBasis!(basis, proj, g)
        @printf("%3i : ω=(%24.8f, %24.8f) -> error=%24.16g\n", basis.N, g[1], g[2], basis.residual[end])
        maxResidual, ωi = findmax(residual)
        plotResidual(basis)
    end
    testOrthgonal(basis)
    @printf("residual = %.10e\n", maxResidual)
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
        
    for qi in 1:basis.N - 1
        q = basis.Q[qi, :]
        qnew -= projqq(basis, q, qnew) .* q  # <q, qnew> q
    end
    return qnew / sqrt(projqq(basis, qnew, qnew))
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

# function Residual(basis, proj, g::Float)
#     # norm2 = proj(g, g) - \sum_i (<qi, K_g>)^2
#     # qi=\sum_j Q_ij K_j ==> (<qi, K_g>)^2 = (\sum_j Q_ij <K_j, K_g>)^2 = \sum_jk Q_ij*Q_ik <K_j, K_g>*<K_k, Kg>
    
#     KK = [proj(basis.Λ, gj, g) for gj in basis.grid]
#     norm2 = proj(basis.Λ, g, g) - (norm(basis.Q * KK))^2
#     return norm2 < 0 ? Float(0) : sqrt(norm2) 
#     end
        
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
    Λ = Float(1e3)
    rtol = Float(1e-8)
    dim = 2
    g0 = [[Float(0), Float(0)], [Float(0), Λ], [Λ, Float(0)], [Λ, Λ]]
    # println(unilog(Λ, rtol))
    # Λ = 100
    @time ωBasis = QR(dim, Λ, rtol, projExp_τ, g0)
    # @time τBasis = QR(Λ / 2, 1e-11, projPHA_τ, Float(0), N=ωBasis.N)
    # nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :acorr)

    # @time basis = QR(100, 1e-10)
    # readline()
    # basis = QR(100, 1e-3)

end