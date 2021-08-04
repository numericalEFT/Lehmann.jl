using LinearAlgebra:Matrix, zero, similar
using LinearAlgebra, Printf
using Roots
using Quadmath
using Plots
# using ProfileView

# const Float = Float64
# const Float = BigFloat
const Float = Float128

include("./kernel.jl")

mutable struct Basis
    Λ::Float
    rtol::Float

    N::Int
    grid::Vector{Float}
    residual::Vector{Float}
    Q::Matrix{Float} # u_j = K_{g_i}*Q_{ij}
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
    insert!(basis.residual, idx, maximum(residual))
# scanResidual!(basis, idx-9, idx+10, g0)
    return idx, candidate, residual
end

function scanResidual!(basis, proj, g0, idx)
    candidate = zeros(Float, basis.N)
    residual = zeros(Float, basis.N)
    for i in 1:basis.N # because of the separation of scales, the grids far away from idx is rarely affected
    # for i in 1:basis.N
        if i < 1 || i > basis.N
            continue
        elseif i == basis.N
            # take care of the last bin
            if g0 < eps(Float(0))
                g = findCandidate(basis, proj, Float(0), Float(10))
            else
                g = findCandidate(basis, proj, basis.grid[end], basis.grid[end] * 10)
            end
            if g > basis.Λ
                candidate[end] = basis.Λ
                residual[end] = Residual(basis, proj, basis.Λ)
            else
                candidate[end] = g
                residual[end] = Residual(basis, proj, g)
            end
        else # 1<=i<=basis.N-1
            g = findCandidate(basis, proj, basis.grid[i], basis.grid[i + 1])
            candidate[i] = g
            residual[i] = Residual(basis, proj, g)
        end
    end
    return candidate, residual
end

function QR(Λ, rtol, proj, isMatfreq; N=nothing)
    basis = Basis(Λ, rtol)
    idx, candidate, residual = addBasis!(basis, proj, Float(0))
    maxResidual, ωi = findmax(residual)

    while isnothing(N) ? maxResidual > rtol : basis.N < N

        newω = candidate[ωi]
        idx, candidate, residual = addBasis!(basis, proj, newω)

        @printf("%3i : ω=%24.8f ∈ (%24.8f, %24.8f) -> error=%24.16g\n", basis.N, newω, basis.grid[idx - 1], (idx == basis.N) ? Λ : basis.grid[idx + 1], basis.residual[idx])
        # println("$(length(freq)) basis: ω=$(Float64(newω)) between ($(Float64(freq[idx - 1])), $(Float64(freq[idx + 1])))")
        
    #     for i in 1:length(candidates)
    #         @printf("%16.8f  ->  %16.8f\n", candidates[i], residual[i])
    # end

        # ω = LinRange(Float(0), Float(100), 1000)
        # y = [Residual(basis, w) for w in ω]
        # p = plot(ω, y, xlims=(0.0, 100))
        # display(p)
        # readline()
    
        maxResidual, ωi = findmax(residual)
    end
    testOrthgonal(basis)
    # @printf("residual = %.10e, Fnorm/F0 = %.10e\n", residual, residualF(freq, Q, Λ))
    @printf("residual = %.10e\n", maxResidual)

    ω = LinRange(Float(0), Float(100), 1000)
    y = [Residual(basis, proj, w) for w in ω]
    p = plot(ω, y, xlims=(0.0, 100))
    plot!(p, candidate, residual, seriestype=:scatter)
display(p)
    readline()

    return basis
end

"""
q1=sum_j c_j K_j
q2=sum_k d_k K_k
return <q1, q2> = sum_jk c_j*d_k <K_j, K_k>
"""
function projqq(basis, q1::Vector{Float}, q2::Vector{Float})
    return q1' * basis.proj * q2
    end
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

    norm = sqrt(projqq(basis, qnew, qnew))
    return qnew / norm
end

"""
Gram-Schmidt process
"""
function GramSchmidt(basis, idx, g::Float)
    q0 = zeros(Float, basis.N)
    q0[idx] = 1
    qnew = copy(q0)
        
    for qi in 1:basis.N
        if qi == idx
    continue
    end
        q = basis.Q[qi, :]
        qnew -=  projqq(basis, q, q0) .* q
    end
    
    norm = sqrt(projqq(basis, qnew, qnew))
    return qnew / norm
end

function Residual(basis, proj, g::Float)
    # norm2 = proj(g, g) - \sum_i (<qi, K_g>)^2
    # qi=\sum_j Q_ij K_j ==> (<qi, K_g>)^2 = (\sum_j Q_ij <K_j, K_g>)^2 = \sum_jk Q_ij*Q_ik <K_j, K_g>*<K_k, Kg>
    
    KK = [proj(basis.Λ, gj, g) for gj in basis.grid]
    norm2 = proj(basis.Λ, g, g) - (norm(basis.Q * KK))^2
    return norm2 < 0 ? Float(0) : sqrt(norm2) 
end
    
function findCandidate(basis, proj, gmin::Float, gmax::Float)
    N = 16
    dg = abs(gmax - gmin) / N
    r0 = Residual(basis, proj, gmin)
    g = gmin + dg
        r = Residual(basis, proj, g)
        if r < r0
       println("warning: $r at $g < $r0 at $gmin  !")

       ω = LinRange(Float(0), Float(10), 1000)
        y = [Residual(basis, proj, w) for w in ω]
       p = plot(ω, y, xlims=(0.0, 10))
       display(p)
       readline()

       exit(0)
    end
    while r > r0
    g += dg
        r0 = r
    r = Residual(basis, proj, g)
    end
    return g - dg
end


function testOrthgonal(basis)
    println("testing orthognalization...")
    II = basis.Q * basis.proj * basis.Q'
    maxerr = maximum(abs.(II - I))
    println("Max Orthognalization Error: ", maxerr)
end
    

if abspath(PROGRAM_FILE) == @__FILE__    
    # freq, Q = findBasis(1.0e-3, Float(100))
    # basis = QR(100, 1e-3)
    Λ = 1000000
    # Λ = 100
    @time ωBasis = QR(Λ, 1e-10, projPH_ω, false)
    @time τBasis = QR(Λ / 2, 1e-11, projPH_τ, false, N=ωBasis.N)
    # @time basis = QR(100, 1e-10)
    # readline()
    # basis = QR(100, 1e-3)

end