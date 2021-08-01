using LinearAlgebra, Printf
using Roots
using Quadmath
# using Plots

# const Float = Float64
const Float = BigFloat
# const Float = Float128

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

function findFreqMax(freq, Q, ωmin, ωmax)
    if DNorm2(freq, Q, ωmin) * DNorm2(freq, Q, ωmax) < Float(0)
        return find_zero(x -> DNorm2(freq, Q, x), (ωmin, ωmax), Bisection(), rtol=1e-5)
    else
        println("warning: $ωmin -> $ωmax derivatives have the same sign $(DNorm2(freq, Q, ωmin)) -> $(DNorm2(freq, Q, ωmax)) !")
        println(DNorm2(freq, Q, ωmin))
        println(DNorm2(freq, Q, ωmax))
        exit(0)
        return sqrt(ωmin * ωmax) # simply return the median of the two frequencies
    end
    end

# plus(ω) = ω + eps(ω) * 1000
    # minus(ω) = ω - eps(ω) * 1000
    plus(ω) = ω + Float(1e-3)
    minus(ω) = ω - Float(1e-3)

    function findBasis(eps, Λ)
    freq = [Float(0), ]
    error = [Float(1), ]
    Q = [[1 / Norm(freq[1]), ], ]
    candidates = [findFreqMax(freq, Q, plus(freq[1]), minus(Λ)), ] # ω with the maximum residual in each segement
    residual = [Norm(freq, Q, candidates[1]), ]
    maxResidual, ωi = residual[1], 1

        while maxResidual > eps
        # println("next : ", candidates[ωi], " -> ", maxResidual)

        newω = candidates[ωi]

        idx = addFreq!(freq, Q, newω)

        if idx < length(freq)
            @printf("%3i : ω=%16.8f ∈ (%16.8f, %16.8f)\n", length(freq), newω, freq[idx - 1], freq[idx + 1])
            # println("$(length(freq)) basis: ω=$(Float64(newω)) between ($(Float64(freq[idx - 1])), $(Float64(freq[idx + 1])))")
        else
            @printf("%3i : ω=%16.8f ∈ (%16.8f, Λ)\n", length(freq), newω, freq[idx - 1])
            # println("$(length(freq)) basis: ω=$(Float64(newω)) for the last freq $(Float64(freq[idx - 1]))")
        end
        
        # if idx == length(freq)
        #     # ωMax = (freq[idx] * 10 > Λ) ? Λ : freq[idx] * 10
        #     ωMax = freq[idx] * 10
        # else
        #     ωMax = freq[idx + 1]
        # end
        r1 = findFreqMax(freq, Q, plus(freq[idx - 1]), minus(freq[idx]))
        ωMax = (idx == length(freq)) ? freq[idx] * 10 : freq[idx + 1]
        r2 = findFreqMax(freq, Q, plus(freq[idx]), minus(ωMax))
        r2 = (r2 > Λ) ? Λ : r2

        candidates[idx - 1] = r1
        insert!(candidates, idx, r2)
        residual[idx - 1] = Norm(freq, Q, r1)
        insert!(residual, idx, Norm(freq, Q, r2))
    
        maxResidual, ωi = findmax(residual)
    end
    testOrthgonal(freq, Q)
    # @printf("residual = %.10e, Fnorm/F0 = %.10e\n", residual, residualF(freq, Q, Λ))
    @printf("residual = %.10e\n", maxResidual)
return freq, Q
end

# function findFreqMax(freq, Q, idx, Λ)
#     if idx == length(freq)
#         # the last segement, bounded by Λ
#         if DNorm2(freq, Q, Λ) > 0.0
#             return Λ
#         end

#         if idx == 1
#             # ∈ (0, Λ)
#             ω = find_zero(x -> DNorm2(freq, Q, x), (Float(1e-6), Float(10)), Bisection(), rtol=1e-5)
#         else
#             maxω = 10 * freq[end] > Λ ? Λ : 100 * freq[end]
#             ω = find_zero(x -> DNorm2(freq, Q, x), (freq[end] * (1 + 1e-3), maxω), Bisection(), rtol=1e-5)
#         end
#         return ω
#     else
#         # println("DNorm2: ", DNorm2(freq, Q, freq[idx] * (2 + 1e-1)), " -> ", DNorm2(freq, Q, freq[idx + 1] * (0.5 - 1e-1)))
#         if idx == 1
#             d1, d2 = Float(1e-6), freq[idx + 1] * (1 - 1e-1)
#         else
#             d1, d2 = freq[idx] * (1 + 1e-6), freq[idx + 1] * (1 - 1e-6)
#         end
#         if sign(DNorm2(freq, Q, d1)) == sign(DNorm2(freq, Q, d2))
#             # println(d1, ", ", d2)
#             println("warning: $(freq[idx]) -> $(freq[idx + 1]) derivatives have the same sign $(DNorm2(freq, Q, d1)) -> $(DNorm2(freq, Q, d1)) !")

#             ω = sqrt(freq[idx] * freq[idx + 1])
#     else
#             ω = find_zero(x -> DNorm2(freq, Q, x), (d1, d2), Bisection(), rtol=1e-6)
#         end
#         return ω
# end
# end

# function findFreqMedian(freq, Q, idx, Λ)
# if idx == length(freq)
    #         return sqrt(freq[idx] * Λ)
# else
#         return sqrt(freq[idx] * freq[idx + 1])
# end
# end

# function scheme1(eps, Λ)
#     freq = Vector{Float}([Float(0), ])
#     Q = [[1 / Norm(freq[1]), ], ]
#     residual = 1.0
#     candidates = [findFreqMax(freq, Q, 1, Λ), ]

#     while residual > eps
#         maxR = Float(0)
#             idx, ifreq, newω = 1, 1, 1
#         for i in 1:length(freq)
#             # ω = findFreqMax(freq, Q, i)
#             ω = candidates[i]
#             normw = Norm(freq, Q, ω)
#             if normw > maxR
#                 maxR = normw
#                 # idx = k + 1
#                 ifreq = i
#         newω = ω 
#             end
#         end
#         residual = maxR
#         # residual = 
#         if residual > eps
#             idx = addFreq!(freq, Q, newω)
#             # println("add $(length(freq))")
#             if idx < length(freq)
#                 @printf("%3i : ω=%16.8f ∈ (%16.8f, %16.8f)\n", length(freq), newω, freq[idx - 1], freq[idx + 1])
#                 # println("$(length(freq)) basis: ω=$(Float64(newω)) between ($(Float64(freq[idx - 1])), $(Float64(freq[idx + 1])))")
#             else
#                 @printf("%3i : ω=%16.8f ∈ (%16.8f, Λ)\n", length(freq), newω, freq[idx - 1])
#                 # println("$(length(freq)) basis: ω=$(Float64(newω)) for the last freq $(Float64(freq[idx - 1]))")
#             end
#             # println("residual=$residual")
#             # @assert newidx == idx "idx: $idx != newidx: $newidx"
#             # testOrthgonal(freq, Q)
            
#             deleteat!(candidates, ifreq)
#             @assert idx > 1 "idx=$idx"
#             push!(candidates, findFreqMax(freq, Q, idx - 1, Λ))
#             push!(candidates, findFreqMax(freq, Q, idx, Λ))
#         end
#     end
#     testOrthgonal(freq, Q)
#     # @printf("residual = %.10e, Fnorm/F0 = %.10e\n", residual, residualF(freq, Q, Λ))
#     @printf("residual = %.10e\n", residual)
# return freq, Q
# end

if abspath(PROGRAM_FILE) == @__FILE__    
    findBasis(1.0e-3, Float(10))

    # ω = LinRange(Float(0), Float(0.5), 1000)
# y = [Norm(freq, Q, w) for w in ω]
    # p = plot(ω, y, xlims=(0.0, 0.5))
    # display(p)
    # readline()
end