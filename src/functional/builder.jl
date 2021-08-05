using LinearAlgebra:Matrix, zero, similar
using LinearAlgebra, Printf
using Roots
using Quadmath
# using ProfileView

# const Float = Float64
# const Float = BigFloat
const Float = Float128

include("./kernel.jl")
include("./matfreq.jl")


using Plots
function plotResidual(basis, proj, gmin, gmax, candidate=nothing, residual=nothing)
    ω = LinRange(gmin, gmax, 1000)
    y = [Residual(basis, proj, w) for w in ω]
    p = plot(ω, y, xlims=(gmin, gmax))
    if isnothing(candidate) == false
        plot!(p, candidate, residual, seriestype=:scatter)
    end
    display(p)
    readline()
end

mutable struct Basis
    Λ::Float
    rtol::Float

    N::Int # number of basis
    grid::Vector{Float} # grid for the basis
    residual::Vector{Float} # achieved error by each basis
    Q::Matrix{Float} # K = Q*R
    proj::Matrix{Float} # the overlap of basis functions <K(g_i), K(g_j)>

    function Basis(Λ, rtol)
        _Q = Matrix{Float}(undef, (0, 0))
        return new(Λ, rtol, 0, [], [], _Q, similar(_Q))
    end
end

function addBasis!(basis, proj, g0::Float)
    basis.N += 1
    if basis.N == 1
        idx = 1
        basis.grid = [g0, ]
        basis.Q = zeros(Float, (basis.N, basis.N))
        basis.Q[1,1] = 1 / sqrt(proj(basis.Λ, g0, g0))
        basis.proj = projKernel(basis, proj)
    else
        idxList = findall(x -> x > g0, basis.grid)
        # if ω is larger than any existing freqencies, then idx is an empty list
        idx = length(idxList) == 0 ? basis.N : idxList[1] # the index to insert the new frequency

        insert!(basis.grid, idx, g0)
        basis.proj = projKernel(basis, proj)
        _Q = copy(basis.Q)
        basis.Q = zeros(Float, (basis.N, basis.N))
        basis.Q[1:idx - 1, 1:idx - 1] = _Q[1:idx - 1, 1:idx - 1]
        basis.Q[1:idx - 1, idx + 1:end] = _Q[1:idx - 1, idx:end]
        basis.Q[idx + 1:end, 1:idx - 1] = _Q[idx:end, 1:idx - 1]
        basis.Q[idx + 1:end, idx + 1:end] = _Q[idx:end, idx:end]
        # println(maximum(abs.(GramSchmidt(basis, idx, g0) .- mGramSchmidt(basis, idx, g0))))
        basis.Q[idx, :] = mGramSchmidt(basis, idx, g0)
    end

    candidate, residual = scanResidual!(basis, proj, g0, idx)
    insert!(basis.residual, idx, maximum(residual)) # record error after the new grid is added
    return idx, candidate, residual
end

function scanResidual!(basis, proj, g0, idx)
    candidate, residual = zeros(Float, basis.N), zeros(Float, basis.N)

    for i in 1:basis.N # because of the separation of scales, the grids far away from idx is rarely affected
        if i == 1
            if basis.grid[1] > Float(0) # if the first grid is not 0
                g = findCandidate(basis, proj, Float(0), basis.grid[1])
            else # if the first grid is 0
                g = Float(0)
            end
        else # 1<=i<=basis.N-1
            # println("between ", basis.grid[i - 1], " , ", basis.grid[i])
            g = findCandidate(basis, proj, basis.grid[i - 1], basis.grid[i])
            # println("got ", g)
        end
        # println("$i -> $g")
        candidate[i] = g
        residual[i] = Residual(basis, proj, g)
    end
    # println(basis.grid)
    # println(candidate)
    # println(residual)
    return candidate, residual
end

function QR(Λ, rtol, proj; N=nothing)
    basis = Basis(Λ, rtol)
    idx, candidate, residual = addBasis!(basis, proj, Float(Λ))
    maxResidual, ωi = findmax(residual)

    while isnothing(N) ? maxResidual > rtol : basis.N < N

        newω = candidate[ωi]
        idx, candidate, residual = addBasis!(basis, proj, newω)

        @printf("%3i : ω=%24.8f ∈ (%24.8f, %24.8f) -> error=%24.16g\n", basis.N, newω, (idx == 1) ? 0 : basis.grid[idx - 1], basis.grid[idx + 1], basis.residual[idx])
        # println("$(length(freq)) basis: ω=$(Float64(newω)) between ($(Float64(freq[idx - 1])), $(Float64(freq[idx + 1])))")
        # plotResidual(basis, proj, Float(0), Float(100), candidate, residual)
        maxResidual, ωi = findmax(residual)
    end
    testOrthgonal(basis)
    # @printf("residual = %.10e, Fnorm/F0 = %.10e\n", residual, residualF(freq, Q, Λ))
    @printf("residual = %.10e\n", maxResidual)
    plotResidual(basis, proj, Float(0), Float(100), candidate, residual)
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
            K[i,j] = proj(basis.Λ, basis.grid[i], basis.grid[j])
        end
    end
    return K
end
    
"""
modified Gram-Schmidt process
"""
function mGramSchmidt(basis, idx, g::Float)
    qnew = zeros(Float, basis.N)
    qnew[idx] = 1
        
    for qi in 1:basis.N
        if qi == idx
            continue
        end
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

function Residual(basis, proj, g::Float)
    # norm2 = proj(g, g) - \sum_i (<qi, K_g>)^2
    # qi=\sum_j Q_ij K_j ==> (<qi, K_g>)^2 = (\sum_j Q_ij <K_j, K_g>)^2 = \sum_jk Q_ij*Q_ik <K_j, K_g>*<K_k, Kg>
    
    KK = [proj(basis.Λ, gj, g) for gj in basis.grid]
    norm2 = proj(basis.Λ, g, g) - (norm(basis.Q * KK))^2
    return norm2 < 0 ? Float(0) : sqrt(norm2) 
end
    
function findCandidate(basis, proj, gmin::Float, gmax::Float)
    if gmin == 0 && gmax > 100  
       gmax = 100 # if the first grid is 0, then the maximum won't be larger than 100
    end
    if gmin > 0 && gmax / gmin > 100
       gmax = gmin * 100  # the maximum won't be larger than 100*gmin
    end

    N = 16
    dg = abs(gmax - gmin) / N
    g = gmin
    r0 = Residual(basis, proj, g)
    r = Residual(basis, proj, g + dg)
    if r < r0 && gmin > Float(0)
        println("warning: $r at $(g + dg) < $r0 at $g  !")
        exit(0)
    end
    while r > r0
        g += dg
        r0 = r
        r = Residual(basis, proj, g + dg)
    end
    if g + dg > gmax
        println("warning: $(g + dg) are not within ($gmin, $gmax)!")
    end
    return g
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
function dlr(type, Λ, rtol)
    println(rtol)
    if type == :corr
        println("Building ω grid ... ")
        ωBasis = QR(Λ, rtol, projPH_ω)
        println("Building τ grid ... ")
        τBasis = QR(Λ / 2, rtol / 10, projPH_τ, N=ωBasis.N)
        println("Building n grid ... ")
        nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :corr)
    elseif type == :acorr
        println("Building ω grid ... ")
        ωBasis = QR(Λ, rtol, projPHA_ω)
        println("Building τ grid ... ")
        τBasis = QR(Λ / 2, rtol / 10, projPHA_τ, N=ωBasis.N)
        println("Building n grid ... ")
        nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :acorr)
    end
    rank = ωBasis.N
    ωGrid = ωBasis.grid
    τGrid = τBasis.grid / Λ
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
    Λ = 100
    # Λ = 100
    @time ωBasis = QR(Λ, 1e-3, projPHA_ω)
    # @time τBasis = QR(Λ / 2, 1e-11, projPHA_τ, N=ωBasis.N)
    # nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :acorr)

    # @time basis = QR(100, 1e-10)
    # readline()
    # basis = QR(100, 1e-3)

end