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
    grids = copy(basis.grid)
    if basis.grid[1] > Float(0)
        insert!(grids, 1, Float(0))
    end
    if basis.grid[end] < basis.Λ
        append!(grids, basis.Λ)
    end
    candidate, residual = zeros(Float, length(grids) - 1), zeros(Float, length(grids) - 1)

    for i in 1:length(grids) - 1 # because of the separation of scales, the grids far away from idx is rarely affected
        g = findCandidate(basis, proj, grids[i], grids[i + 1])
        candidate[i] = g
        residual[i] = Residual(basis, proj, g)
    end
    return candidate, residual
end

function QR(Λ, rtol, proj, g0; N=nothing)
    basis = Basis(Λ, rtol)
    idx, candidate, residual = addBasis!(basis, proj, Float(g0))
    @printf("%3i : ω=%24.8f ∈ (%24.8f, %24.8f) -> error=%24.16g\n", 1, g0, 0, Λ, basis.residual[idx])
    maxResidual, ωi = findmax(residual)

    while isnothing(N) ? maxResidual > rtol/10 : basis.N < N

        newω = candidate[ωi]
        idx, candidate, residual = addBasis!(basis, proj, newω)
        # println(length(basis.grid))
        # println(idx)
        lower = (idx == 1) ? 0 : basis.grid[idx - 1]
        upper = (idx == basis.N) ? Λ : basis.grid[idx + 1]

        @printf("%3i : ω=%24.8f ∈ (%24.8f, %24.8f) -> error=%24.16g\n", basis.N, newω, lower, upper, basis.residual[idx])
        # println("$(length(freq)) basis: ω=$(Float64(newω)) between ($(Float64(freq[idx - 1])), $(Float64(freq[idx + 1])))")
        # plotResidual(basis, proj, Float(0), Float(100), candidate, residual)
        maxResidual, ωi = findmax(residual)
    end
    testOrthgonal(basis)
    # @printf("residual = %.10e, Fnorm/F0 = %.10e\n", residual, residualF(freq, Q, Λ))
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
    @assert gmax > gmin

    if abs(gmin) < 100 * eps(Float(0)) && gmax > 100  
       gmax = 100 # if the first grid is 0, then the maximum should be between (0, 100)
    end
    if abs(gmin) > 100 * eps(Float(0)) && gmax / gmin > 100
       gmax = gmin * 100  # the maximum won't be larger than 100*gmin
    end

    N = 32
    dg = (gmax - gmin) / N

    ###################   if gmin/gmax are at the boundary 0/Λ, the maximum could be at the edge ##################
    if abs(gmin) < 100 * eps(Float(0)) && Residual(basis, proj, gmin) > Residual(basis, proj, gmin + dg)
        return gmin
    end
    if abs(gmax - basis.Λ) < 100 * eps(Float(gmax)) && Residual(basis, proj, gmax) > Residual(basis, proj, gmax - dg)
        return gmax
    end

    ###################  the maximum must be between (gmin, gmax) for the remaining cases ##################
    # check https://www.geeksforgeeks.org/find-the-maximum-element-in-an-array-which-is-first-increasing-and-then-decreasing/ for detail

    l, r = 1, N-1 #avoid the boundary gmin and gmax
    while l<=r
        m = l + Int(round((r - l) / 2))
        g = gmin+m*dg

        r1, r2, r3 = Residual(basis, proj, g-dg), Residual(basis, proj, g), Residual(basis, proj, g + dg)
        if r2 >= r1 && r2 >= r3
            # plotResidual(basis, proj, gmin, gmax)
            return g
        end

        if r3 < r2 < r1
            r = m - 1
        elseif r1<r2<r3
            l = m + 1
        else
            if abs(r1 - r2)<1e-17 && abs(r2-r3)<1e-17
                return g
            end
            println("warning: illegl! ($l, $m, $r) with ($r1, $r2, $r3)")
            plotResidual(basis, proj, gmin, gmax)
            exit(0)
        end
    end
    # plotResidual(basis, proj, gmin, gmax)
    throw("failed to find maximum between ($gmin, $gmax)!")


    # N = 16
    # dg = (gmax - gmin) / N
    # g = gmin
    # r0 = Residual(basis, proj, g)
    # r = Residual(basis, proj, g + dg)
    # while r > r0 
    #     g += dg
    #     if abs(g - gmax) < 100 * eps(Float(gmax))
    #         break
    #     end
    #     r0 = r
    #     r = Residual(basis, proj, g + dg)
    # end

    # if abs(g - gmax) < 100 * eps(Float(gmax)) && gmax < basis.Λ # at the upper boundary, gmax could be the maximum
    #     println("warning: $g touch the upper bound $gmax!")
    #     # plotResidual(basis, proj, gmin, gmax)
    # end
    # if abs(g - gmin) < 100 * eps(Float(gmin)) && gmin > 0 # at the lower boundary, gmin could be the maximum
    #             println("warning: $g touch the lower bound $gmin!")
    #     # plotResidual(basis, proj, gmin, gmax)
    # end
    # return g
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
    Λ = Float(Λ)
    if type == :corr
        println("Building ω grid ... ")
        ωBasis = QR(Λ, rtol, projPH_ω, Λ)
        println("Building τ grid ... ")
        τBasis = QR(Λ / 2, rtol / 10, projPH_τ, Float(0), N=ωBasis.N)
        println("Building n grid ... ")
        nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :corr)
    elseif type == :acorr
        println("Building ω grid ... ")
        ωBasis = QR(Λ, rtol, projPHA_ω, Λ)
        println("Building τ grid ... ")
        τBasis = QR(Λ / 2, rtol / 10, projPHA_τ, Float(0), N=ωBasis.N)
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
    Λ = 1e12
    # Λ = 100
    @time ωBasis = QR(Λ, 1e-10, projPHA_ω, Λ)
    @time τBasis = QR(Λ / 2, 1e-11, projPHA_τ, Float(0), N=ωBasis.N)
    # nBasis = MatFreqGrid(ωBasis.grid, ωBasis.N, Λ, :acorr)

    # @time basis = QR(100, 1e-10)
    # readline()
    # basis = QR(100, 1e-3)

end