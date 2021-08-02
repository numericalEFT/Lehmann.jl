using LinearAlgebra:Matrix, zero
using LinearAlgebra, Printf
using Roots
using Quadmath
using Plots

const Float = Float64
# const Float = BigFloat
# const Float = Float128

mutable struct Basis
    Λ::Float
    rtol::Float

    N::Int
    grid::Vector{Float}
    Q::Matrix{Float} # u_j = K_{g_i}*Q_{ij}
    residual::Vector{Float}
    candidate::Vector{Float}
    function Basis(Λ, rtol)
        _Q = Matrix{Float}(undef, (0, 0))
        return new(Λ, rtol, 0, [], _Q, [], [])
    end
    # function Basis(g0::Float)
    #     _g = Vector{Float}(g0)
    #     _Q = Matix{Float}(undef, 1, 1)
    #     _Q[1, 1] = 1 / 
    #     return Basis(_g, _r, _Q)
    # end
    # function Basis(_g, _r, _Q)
    #     @assert length(grid) = size(Q)[1] == size(Q)[2] == length(_r)
    #     return Basis(grid, residual, Q)
    # end
end

function addBasis(basis, g0::Float)
    if basis.N == 0
        idx = 1
        qnew = [1 / Norm(g0), ]
    else
        idxList = findall(x -> x > g0, basis.grid)
        # if ω is larger than any existing freqencies, then idx is an empty list
        idx = length(idxList) == 0 ? basis.N + 1 : idxList[1] # the index to insert the new frequency
        # idx = length(freq) + 1 # always add the new freq to the end

        qnew, q00 = orthognalize(basis, g0)
        insert!(qnew, idx, q00)
    end
    basis.N += 1
    insert!(basis.grid, idx, g0)
    # add a new column and a new row for the new grid point
    _Q = copy(basis.Q)
    basis.Q = zeros(Float, (basis.N, basis.N))
    basis.Q[1:idx - 1, 1:idx - 1] = _Q[1:idx - 1, 1:idx - 1]
    basis.Q[1:idx - 1, idx + 1:end] = _Q[1:idx - 1, idx:end]
    basis.Q[idx + 1:end, 1:idx - 1] = _Q[idx:end, 1:idx - 1]
    basis.Q[idx + 1:end, idx + 1:end] = _Q[idx + 1:end, idx + 1:end]
    basis.Q[idx, :] = qnew
    println(basis.grid)
    println(basis.Q)

    testOrthgonal(basis)
    ω = LinRange(Float(0), Float(10), 1000)
    y = [residual(basis, w) for w in ω]
    p = plot(ω, y, xlims=(0.0, 10))
    display(p)
    readline()


    basis.residual = zeros(Float, basis.N)
    basis.candidate = zeros(Float, basis.N)
        
    for i in 1:basis.N - 1
        g = findCandidate(basis, basis.grid[i], basis.grid[i + 1])
        basis.candidate[i] = g
        basis.residual[i] = residual(basis, g)
    end
    
    if g0 < eps(Float(0))
        g = findCandidate(basis, Float(0), Float(10))
    else
        g = findCandidate(basis, basis.grid[end], basis.grid[end] * 10)
    end
    if g > basis.Λ
        basis.candidate[end] = basis.Λ
        basis.residual[end] = 0
    else
        basis.candidate[end] = g
        basis.residual[end] = residual(basis, g)
    end

    return idx, basis
end

function QR(Λ, rtol)
    basis = Basis(Λ, rtol)
    addBasis(basis, Float(0))
    maxResidual, ωi = findmax(basis.residual)

    while maxResidual > rtol

        newω = basis.candidate[ωi]
        idx, basis = addBasis(basis, newω)

        @printf("%3i : ω=%16.8f ∈ (%16.8f, %16.8f)\n", basis.N, newω, basis.grid[idx - 1], (idx == basis.N) ? Λ : basis.grid[idx + 1])
        # println("$(length(freq)) basis: ω=$(Float64(newω)) between ($(Float64(freq[idx - 1])), $(Float64(freq[idx + 1])))")
        
    #     for i in 1:length(candidates)
    #         @printf("%16.8f  ->  %16.8f\n", candidates[i], residual[i])
    # end

    #     ω = LinRange(Float(0), Float(10), 1000)
#     y = [Norm(freq, Q, w) for w in ω]
    #     p = plot(ω, y, xlims=(0.0, 10))
    #     display(p)
    #     readline()
    
        maxResidual, ωi = findmax(basis.residual)
    end
    # testOrthgonal(freq, Q)
    # @printf("residual = %.10e, Fnorm/F0 = %.10e\n", residual, residualF(freq, Q, Λ))
    @printf("residual = %.10e\n", maxResidual)
    return basis
end

"""
calculate the new basis projected to the existing orthonormal basis
return [<new basis, Q[:, i]>  for i ∈ 1:N ]
"""
function proj(basis, g::Float)
    proj_j = [proj(gj, g) for gj in basis.grid]
    return basis.Q*proj_j
end

function orthognalize(basis, g::Float)
    qnew = proj(basis, g) .*(-1)
    return qnew, 1 / residual(basis, g)
end

function residual(basis, g::Float)
    norm2 = proj(g, g) - sum((proj(basis, g)).^2)
    return sqrt(abs(norm2)) # norm2 may become slightly negative if ω concides with the existing frequencies
    end
    
function findCandidate(basis, gmin::Float, gmax::Float)
    N = 16
    dg = abs(gmax - gmin) / N
    r0 = residual(basis, gmin)
    g = gmin + dg
    r = residual(basis, g)
    if r <= r0
       println("warning: $r at $g < $r0 at $gmin  !")

    #    ω = LinRange(Float(0), Float(gmax), 1000)
    #    y = [residual(basis, w) for w in ω]
    #    p = plot(ω, y, xlims=(0.0, 10))
    #    display(p)
    #    readline()

       exit(0)
    end
    while r > r0
        g += dg
        r0 = r
        r = residual(basis, g)
    end
    return g - dg
end


function testOrthgonal(basis)
    println("testing orthognalization...")
    II = zeros(Float, (basis.N, basis.N))
    K = zeros(Float, (basis.N, basis.N))
    for i in 1:basis.N
        for j in 1:basis.N
            K[i,j] = proj(basis.grid[i], basis.grid[j])
        end
    end
    II = basis.Q*K*basis.Q'
    maxerr = maximum(abs.(II - I))
    println("Max Orthognalization Error: ", maxerr)
# @assert maxerr < atol
end
    

"""
\\int_0^1 e^{-ω_1 τ}*e^{-ω_2*τ} dτ = (1-exp(-(ω_1+ω_2))/(ω_1+ω_2)
"""
function proj(ω1::Float, ω2::Float)
    ω = ω1 + ω2
    if ω < 1e-6
        return 1 - ω / 2 + ω^2 / 6 - ω^3 / 24 + ω^4 / 120 - ω^5 / 720
    else
    return (1 - exp(-ω)) / ω
    end
end

"""
 derivative for proj(ω, ω)
"""
function dproj(ω::Float)
    if ω < 1e-4
return -1 + 4ω / 3 - ω^2 + 8 * ω^3 / 15 - 2 * ω^4 / 9 + 8 * ω^5 / 105
    else
        return -(1 - exp(-2 * ω)) / (2 * ω^2) + exp(-2ω) / ω
    end
        end

"""
 derivative for proj(ω1, ω2)
"""
function dproj1(ω1::Float, ω2::Float)
    ω = ω1 + ω2
    if ω < 1e-4
return -1 / 2 + ω / 3 - ω^2 / 8 + ω^3 / 30 - ω^4 / 144 + ω^5 / 840
    else
        return -(1 - exp(-ω)) / ω^2 + exp(-ω) / ω
end
end

"""
q=sum_j c_j e^{-ω_j*τ}
return <q, e^{-ωτ}>
"""
function proj(freq, q, ω::Float)
    println(freq)
    println(q)
    @assert length(freq) == length(q)
    return sum([q[i] * proj(freq[i], ω) for i in 1:length(freq)])
end
    
"""
q1=sum_j c_j e^{-ω_j*τ}
q2=sum_k d_k e^{-ω_k*τ}
return <q1, q2> = sum_k d_k <q1, e^{-ω_k*τ}>
"""
function proj(freq, q1::Vector{Float}, q2::Vector{Float})
    @assert length(freq) == length(q1) == length(q2)
    return sum([q2[i] * proj(freq, q1, freq[i]) for i in 1:length(freq)])
end

Norm(ω) = sqrt(abs(proj(ω, ω)))

"""
   qi=sum_j c_ij e^{-ω_j*τ}
   norm2 = <e^{-ω*τ}, e^{-ω*τ}>- \\sum_i c_ij*c_ik*<e^{-ω*τ}, e^{-ω_j*τ}>*<e^{-ω*τ}, e^{-ω_k*τ}>
   return sqrt(norm2)
"""
function Norm(freq, Q, ω::Float)
norm2 = proj(ω, ω) - sum([(proj(freq, q, ω))^2 for q in Q])
    return sqrt(abs(norm2)) # norm2 may become slightly negative if ω concides with the existing frequencies
end
    
            """
First derivative of norm2 for the vector: e^{-ωτ}-sum_i <q_i, e^{-ωτ}>
"""
function DNorm2(freq, Q, ω::Float)
    # d Norm(freq, Q, ω)/dω
    dnorm2 = dproj(ω)
    for j in 1:length(Q)
        for k in 1:length(Q)
            amp = sum([q[j] * q[k] for q in Q])
            dnorm2 -= amp * (dproj1(ω, freq[j]) * proj(ω, freq[k]) + proj(ω, freq[j]) * dproj1(ω, freq[k]))
    end
    end
    return dnorm2 / (2 * Norm(freq, Q, ω))
end

"""
Project the kernel to the DLR basis
"""
function projectedKernel(freq, Q, ω::Float)
    # K(τ, ω) ≈ \sum_i <e^{-ωτ}, qi> qi = \sum_k (\sum_ij c_ij*c_ik <e^{-ωτ, e^{-ωj*τ}}> e^{-ω_k*τ})
    amp = zeros(Float, length(Q))
        for k in 1:length(Q)
        amp[k] = Float(0)
        for i in 1:length(Q)
            for j in 1:length(Q)
                amp[k] += Q[i][j] * Q[i][k] * proj(ω, freq[j])
        end
    end
end
    return amp
end

"""
Calculate the orthnormal vector of the new frequency ω. 
Return the vector q (the projection of the new frequency basis to that of the existing frequencies) and the normalization factor for the new frequency basis.
"""
function orthognalize(freq, Q, ω::Float)
    idx = length(Q) + 1
    qnew = zeros(Float, length(freq))
    
    for (qi, q) in enumerate(Q)
        qnew .-= proj(freq, q, ω) .* q
    end
    
    norm = Norm(freq, Q, ω)
qnew /= norm
    
    return qnew, 1 / norm
end

function testOrthgonal(freq, Q)
    println("testing orthognalization...")
    err = zeros(Float, (length(Q), length(Q)))
for (i, qi) in enumerate(Q)
        for (j, qj) in enumerate(Q)
            err[i, j] = proj(freq, qi, qj)
        end
    end
    maxerr = maximum(abs.(err - I))
    println("Max Orthognalization Error: ", maxerr)
# @assert maxerr < atol
end

# function residualF(freq, Q, Λ)
# #   qi=sum_j c_ij e^{-ω_j*τ}
# #   int_1^\Lambda dω 1/2ω- \sum_i c_ij*c_ik/(ω+ω_j)/(ω+ω_k)
# #    ln(ω)/2- int_1^\Lambda dω \sum_i c_ij*c_ik/(ω+ω_j)/(ω+ω_k)
#     F0 = log(Λ) / 2
#     F = F0
#     # println("omega:", ω, ", ", proj(ω, ω))
#         for j in 1:length(Q)
#         for k in 1:length(Q)
#             amp = Float(0)
#             for i in 1:length(Q)
#                 amp += Q[i][j] * Q[i][k]
#             end
#             if j == k
#                 F -= amp * (1 / (freq[j] + 1) - 1 / (freq[j] + Λ))  
#             else
#                 F -= amp / (freq[k] - freq[j]) * log((freq[j] + Λ) * (freq[k] + 1) / (freq[j] + 1) / (freq[k] + Λ))
#             end
#         end
#     end
    #     return sqrt(F / F0)
# end

"""
add new frequency ω into the existing basis
"""
    function addFreq!(freq, Q, ω)
    idxList = findall(x -> x > ω, freq)
        # if ω is larger than any existing freqencies, then idx is an empty list
    idx = length(idxList) == 0 ? length(freq) + 1 : idxList[1] # the index to insert the new frequency
    # idx = length(freq) + 1 # always add the new freq to the end

    qnew, q00 = orthognalize(freq, Q, ω)
    insert!(freq, idx, ω)
        insert!(Q, idx, qnew)
    for q in Q
        insert!(q, idx, Float(0))  # append zero to the new frequency index in the existing q vectors
    end
    Q[idx][idx] = q00  # add the diagonal element for the current freq
        return idx
    end
    
# function findFreqMax(freq, Q, ωmin, ωmax)
#     dω = abs(ωmax - ωmin) / 10000
#     ωmin += dω
#     ωmax -= dω
#     if DNorm2(freq, Q, ωmin) * DNorm2(freq, Q, ωmax) < Float(0)
#         return find_zero(x -> DNorm2(freq, Q, x), (ωmin, ωmax), Bisection(), rtol=1e-3)
#     else
#         println("warning: $ωmin -> $ωmax derivatives have the same sign $(DNorm2(freq, Q, ωmin)) -> $(DNorm2(freq, Q, ωmax)) !")
#         println(DNorm2(freq, Q, ωmin))
#         println(DNorm2(freq, Q, ωmax))
#         # ω = LinRange(Float(ωmin), Float(ωmax), 1000)
#         # y = [DNorm2(freq, Q, w) for w in ω]
    #         # p = plot(ω, y, xlims=(ωmin, ωmax))
#         # display(p)
#         # readline()
#         # exit(0)
    #         return sqrt(ωmin * ωmax) # simply return the median of the two frequencies
#     end
# end
    
function findFreqMax(freq, Q, ωmin, ωmax)
    N = 16
    dω = abs(ωmax - ωmin) / 16
    r0 = Norm(freq, Q, ωmin)
    ω = ωmin + dω
    r = Norm(freq, Q, ω)
    while r > r0
        ω += dω
        r0 = r
        r = Norm(freq, Q, ω)
    end
    return ω
end

# plus(ω) = ω + Float(1e-3)
# minus(ω) = ω - Float(1e-3)

function findBasis(eps, Λ)
    freq = [Float(0), ]
    error = [Float(1), ]
    Q = [[1 / Norm(freq[1]), ], ]
    # candidates = [findFreqMax(freq, Q, freq[1], Λ), ] # ω with the maximum residual in each segement
    candidates = [findFreqMax(freq, Q, freq[1] + Float(1e-3), Float(10)), ] # ω with the maximum residual in each segement
    residual = [Norm(freq, Q, candidates[1]), ]

        # ω = LinRange(Float(0), Float(10), 1000)
        # y = [DNorm2(freq, Q, w) for w in ω]
        # p = plot(ω, y, xlims=(0.0, 10))
        # display(p)
        # readline()
    
    maxResidual, ωi = residual[1], 1

    while maxResidual > eps
    #     for i in 1:length(candidates)
    #         @printf("%16.8f  ->  %16.8f\n", candidates[i], residual[i])
    # end

        newω = candidates[ωi]
        idx = addFreq!(freq, Q, newω)

        @printf("%3i : ω=%16.8f ∈ (%16.8f, %16.8f)\n", length(freq), newω, freq[idx - 1], (idx == length(freq)) ? Λ : freq[idx + 1])
        # println("$(length(freq)) basis: ω=$(Float64(newω)) between ($(Float64(freq[idx - 1])), $(Float64(freq[idx + 1])))")
        
        r1 = findFreqMax(freq, Q, freq[idx - 1], freq[idx])
        ωMax = (idx == length(freq)) ? freq[idx] * 10 : freq[idx + 1]
        r2 = findFreqMax(freq, Q, freq[idx], ωMax)
        r2 = (r2 > Λ) ? Λ : r2

        candidates[idx - 1] = r1
        # residual[idx - 1] = Norm(freq, Q, r1)
            if newω < Λ
            insert!(candidates, idx, r2)
            # insert!(residual, idx, Norm(freq, Q, r2))
    end
        residual = [Norm(freq, Q, ω) for ω in candidates]

    #     for i in 1:length(candidates)
    #         @printf("%16.8f  ->  %16.8f\n", candidates[i], residual[i])
    # end

    #     ω = LinRange(Float(0), Float(10), 1000)
#     y = [Norm(freq, Q, w) for w in ω]
    #     p = plot(ω, y, xlims=(0.0, 10))
    #     display(p)
    #     readline()
    
        maxResidual, ωi = findmax(residual)
    end
    testOrthgonal(freq, Q)
    # @printf("residual = %.10e, Fnorm/F0 = %.10e\n", residual, residualF(freq, Q, Λ))
    @printf("residual = %.10e\n", maxResidual)
return freq, Q
end

if abspath(PROGRAM_FILE) == @__FILE__    
    # freq, Q = findBasis(1.0e-10, Float(100000000))
    basis = QR(100, 1e-3)

    # ω = LinRange(Float(0), Float(10), 1000)
    # y = [Norm(freq, Q, w) for w in ω]
    # p = plot(ω, y, xlims=(0.0, 10))
    # display(p)
    # readline()
end