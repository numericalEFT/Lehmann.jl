"""
(1-exp(-Λ*x)/x
"""
function kernel(ω::Float)
    if ω < 1e-5
        return 1 - ω / 2 + ω^2 / 6 - ω^3 / 24 + ω^4 / 120 - ω^5 / 720
    else
        # return (1 - exp(-ω)) / ω
        return -expm1(-ω) / ω
    end
end

##################### Particle-hole symmetric kernel #############################

"""
particle-hole symmetric kernel: K(ω, τ)=e^{-ω*τ}+e^{-ω*(β-τ)}

KK=int_0^{1/2} dτ K(ω1,τ)*K(ω2,τ)=(1-e^{ω1+ω2})/(ω1+ω2)+(e^{-ω2}-e^{-ω1})/(ω1-ω2)
"""
function projPH_ω(Λ::Float, ω1::Float, ω2::Float)
    if ω1 > ω2
        return kernel(ω1 + ω2) + exp(-ω2) * kernel(ω1 - ω2)
    else
        return kernel(ω1 + ω2) + exp(-ω1) * kernel(ω2 - ω1)
    end
end

"""
particle-hole symmetric kernel: K(ω, τ)=e^{-ω*τ}+e^{-ω*(β-τ)}

KK=int_0^{Λ} dτ K(ω,t1)*K(ω2,t2)=(1-e^{t1+t2})/(t1+t2)+(1-e^{2β-t1-t2})/(2β-t1-t2)+(1-e^{β+t1-t2})/(β+t1-t2)+(1-e^{β-t1+t2})/(β-t1+t2)
"""
function projPH_τ(Λ::Float, t1::Float, t2::Float)
    # note that here Λ = \beta/2
    return kernel(t1 + t2) + kernel(4 * Λ - t1 - t2) + kernel(2 * Λ - t1 + t2) + kernel(2 * Λ + t1 - t2)
end

##################### Particle-hole asymmetric kernel #############################
"""
particle=hole asymmetric kernel: K(ω, τ)=e^{-ω*τ}-e^{-ω*(β-τ)}

KK=int_0^{1/2} dτ K(ω1,τ)*K(ω2,τ)=(1-e^{ω1+ω2})/(ω1+ω2)-(e^{-ω2}-e^{-ω1})/(ω1-ω2)
"""
function projPHA_ω(Λ::Float, ω1::Float, ω2::Float)
    if ω1 > ω2
        return kernel(ω1 + ω2) - exp(-ω2) * kernel(ω1 - ω2)
    else
        return kernel(ω1 + ω2) - exp(-ω1) * kernel(ω2 - ω1)
    end
end

"""
particle-hole asymmetric kernel: K(ω, τ)=e^{-ω*τ}-e^{-ω*(β-τ)}

KK=int_0^{Λ} dτ K(ω,t1)*K(ω2,t2)=(1-e^{t1+t2})/(t1+t2)+(1-e^{2β-t1-t2})/(2β-t1-t2)-(1-e^{β+t1-t2})/(β+t1-t2)-(1-e^{β-t1+t2})/(β-t1+t2)
"""
function projPHA_τ(Λ::Float, t1::Float, t2::Float)
    return kernel(t1 + t2) + kernel(4 * Λ - t1 - t2) - kernel(2 * Λ - t1 + t2) - kernel(2 * Λ + t1 - t2)
end

"""
inner product of exp(-g1[1]*x1-g1[2]*x2) and exp(-g2[1]*x1-g2[2]*x2),
where x1 ∈ [0.0, 1.0], and x2 ∈ [0.0, x1]
! we assume that x1 and x2 are discretized on a fine grid, with a separation ~ 1 between the grid points
"""
function projExp_τ(Λ::T, dim, g1, g2) where {T}
    # println(g1, ",  ", g2)
    tiny = T(1e-5)
    ω1, ω2 = g1[1] + g2[1], g1[2] + g2[2]
    if ω1 < tiny && ω2 < tiny
        return T(1) / 2
    elseif ω1 < tiny && ω2 > tiny
        return (1 - ω2 - exp(-ω2)) / ω2 / (ω1 - ω2)
    elseif ω1 > tiny && ω2 < tiny
        return (1 - ω1 - exp(-ω1)) / ω1 / (ω2 - ω1)
    elseif abs(ω1 - ω2) < tiny
        ω = (ω1 + ω2) / 2
        return T((1 - exp(-ω) * (1 + ω)) / ω^2)
    else
        return T((ω1 - ω2 + exp(-ω1) * ω2 - exp(-ω2) * ω1) / (ω1 * ω2 * (ω1 - ω2)))
    end
    # if ω1 > tiny && ω2 > tiny
    #     return T((ω1 - ω2 + exp(-ω1) * ω2 - exp(-ω2) * ω1) / (ω1 * ω2 * (ω1 - ω2)))
    # elseif ω1 > tiny && ω2 <= tiny
    # elseif ω2 > tiny && ω1 <= tiny
    # else
    #     return T(0.5) - (ω1 + ω2) / 6 + (ω1^2 + ω1 * ω2 + ω2^2) / 24 - (ω1 + ω2) * (ω1^2 + ω2^2) / 120 + (ω1^4 + ω1^3 * ω2 + ω1^2 * ω2^2 + ω1 * ω2^3 + ω2^4) / 720
    # end

    # if ω1 < tiny || ω2 < tiny
    #     return T(0.5)
    # elseif abs(ω1 - ω2) < tiny
    #     ω = (ω1 + ω2) / 2
    #     return T((1 - exp(-ω) * (1 + ω)) / ω^2)
    # else
    #     return T((ω1 - ω2 + exp(-ω1) * ω2 - exp(-ω2) * ω1) / (ω1 * ω2 * (ω1 - ω2)))
    # end
end

function projExp_τ(Λ::T, dim, g1, g2, gidx1, gidx2, cache) where {T}
    tiny = T(1e-5)
    ω1, ω2 = g1[1] + g2[1], g1[2] + g2[2]
    expω1 = cache[gidx1[1]] * cache[gidx2[1]]
    expω2 = cache[gidx1[2]] * cache[gidx2[2]]
    if ω1 < tiny && ω2 < tiny
        return T(1) / 2
    elseif ω1 < tiny && ω2 > tiny
        return (1 - ω2 - expω2) / ω2 / (ω1 - ω2)
    elseif ω1 > tiny && ω2 < tiny
        return (1 - ω1 - expω1) / ω1 / (ω2 - ω1)
    elseif abs(ω1 - ω2) < tiny
        @assert abs(ω1 - ω2) < eps(Float(1)) * 100 "$ω1 - $ω2 = $(ω1-ω2)"
        return T((1 - expω1 * (1 + ω1)) / ω1^2)
    else
        return T((ω1 - ω2 + expω1 * ω2 - expω2 * ω1) / (ω1 * ω2 * (ω1 - ω2)))
    end
    # if ω1 > tiny && ω2 > tiny
    #     return T((ω1 - ω2 + exp(-ω1) * ω2 - exp(-ω2) * ω1) / (ω1 * ω2 * (ω1 - ω2)))
    # elseif ω1 > tiny && ω2 <= tiny
    # elseif ω2 > tiny && ω1 <= tiny
    # else
    #     return T(0.5) - (ω1 + ω2) / 6 + (ω1^2 + ω1 * ω2 + ω2^2) / 24 - (ω1 + ω2) * (ω1^2 + ω2^2) / 120 + (ω1^4 + ω1^3 * ω2 + ω1^2 * ω2^2 + ω1 * ω2^3 + ω2^4) / 720
    # end

    # if ω1 < tiny || ω2 < tiny
    #     return T(0.5)
    # elseif abs(ω1 - ω2) < tiny
    #     ω = (ω1 + ω2) / 2
    #     return T((1 - exp(-ω) * (1 + ω)) / ω^2)
    # else
    #     return T((ω1 - ω2 + exp(-ω1) * ω2 - exp(-ω2) * ω1) / (ω1 * ω2 * (ω1 - ω2)))
    # end
end